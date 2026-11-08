// mpv-vr: minimal VR test player for flat (2D) video.
//
// The video is shown on a fixed quad ~4 m in front of the viewer. mpv renders
// into ONE shared OpenXR swapchain texture (both eyes see the same image; the
// quad layer supplies the stereo) via its OpenGL render API, no VO window.
//
// Stack: Wayland + EGL + OpenXR (Monado XR_MNDX_egl_enable) + mpv.
// We build a headless EGL/GL context off the wl_display and hand it to
// Monado, which exposes the swapchain images as GL textures we render into.

// XrSwapchainImageOpenGLKHR needs the OpenGL graphics-API guard; Monado's
// EGL binding (XrGraphicsBindingEGLMNDX) needs the EGL platform guard.
#define XR_USE_GRAPHICS_API_OPENGL
#define XR_USE_PLATFORM_EGL
// We call extension entry points (e.g. xrGetOpenGLGraphicsRequirementsKHR),
// so the extension prototypes must be declared.
#define XR_EXTENSION_PROTOTYPES

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <wayland-client.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr.h>

// Core GL 3.0 entry points (glext.h hides them behind version macros).
extern "C" {
GLenum glCheckFramebufferStatus(GLenum target);
void glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers);
void glGenFramebuffers(GLsizei n, GLuint *framebuffers);
void glBindFramebuffer(GLenum target, GLuint framebuffer);
void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget,
                            GLuint texture, GLint level);
GLboolean glIsTexture(GLuint texture);
GLenum glGetError(void);
}

#include <mpv/client.h>
#include <mpv/render.h>
#include <mpv/render_gl.h>

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>

#ifndef GL_SRGB8_ALPHA8
#define GL_SRGB8_ALPHA8 0x8C43
#endif

#define QUAD_Z (-4.0f) // quad sits this far in front of the initial head pose

[[noreturn]] static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "mpv-vr: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

#define XR_CHECK(expr)                                                        \
    do {                                                                      \
        XrResult _r = (expr);                                                 \
        if (_r != XR_SUCCESS)                                                 \
            die("%s -> %d (0x%x)", #expr, (int)_r, (unsigned)_r);             \
    } while (0)

struct App {
    // Wayland + a headless EGL/GL context (handed to Monado for the swapchains)
    wl_display *wl = nullptr;
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLConfig egl_config = nullptr;
    EGLContext egl_context = EGL_NO_CONTEXT;

    // OpenXR
    XrInstance instance{};
    XrSystemId systemId{};
    XrSession session{};
    XrSpace viewSpace{};
    uint32_t viewCount = 0;
    uint32_t width = 0, height = 0;
    // GL internal format of the swapchain textures (e.g. GL_RGBA8). Passed to
    // mpv as the FBO's internal_format so it doesn't have to guess.
    int64_t format = 0;
    std::vector<XrSwapchain> swapchains;
    // Concrete type (24 B), not the 16 B base header: the runtime writes the
    // full XrSwapchainImageOpenGLKHR per element, so the buffer must match.
    std::vector<std::vector<XrSwapchainImageOpenGLKHR>> images;
    std::vector<std::vector<GLuint>> fbos; // [swapchain][image index]

    // mpv
    mpv_handle *mpv = nullptr;
    mpv_render_context *render = nullptr;
    // Value for mpv's "hwdec" option. A comma-separated list is tried in
    // order: zero-copy NVDEC first, then NVDEC with a copy back to system
    // memory. mpv falls back to software decoding if neither can handle the
    // stream. Override with --hwdec=<value> (e.g. --hwdec=no).
    std::string hwdec = "nvdec,nvdec-copy";

    bool quit = false;
    // Mode flags (exactly one is true after argument parsing).
    // --solid: diagnostic; clear the swapchain to flat turquoise, no mpv
    //   (confirms the XR/swapchain/FBO/composition path on its own).
    // --flat: render the video with mpv, flat on the quad (the real 2D path).
    bool solid = false;
    bool flat = false;
    // Frames where the head pose was invalid and the layer was skipped.
    long skipped = 0;
    // Frames where a valid pose was present and the layer was submitted.
    long submitted = 0;
};

// mpv's get_proc_address: resolve GL entry points for this EGL context.
// Behave like eglGetProcAddress (the correct source for an EGL context), with
// a process-symbol-table fallback.
static void *gl_get_proc(void *, const char *name)
{
    void *p = reinterpret_cast<void *>(eglGetProcAddress(name));
    return p ? p : dlsym(RTLD_DEFAULT, name);
}

// Diagnostic: print mpv's playback state so we can tell "no video loaded"
// (idle, tpos stuck at -1) from "video playing" (tpos advancing, size set).
static void print_mpv_status(App &app)
{
    if (!app.mpv)
        return;
    int core_idle = -1;
    double tpos = -1, dur = -1;
    int64_t w = 0, h = 0;
    char *vo = nullptr;
    char *hw = nullptr;
    // Print the mpv_error return codes: a nonzero code means the
    // get_property call itself failed, as opposed to the property simply
    // being "no video" (which returns success with a zero/default value).
    int e_ci = mpv_get_property(app.mpv, "core-idle", MPV_FORMAT_FLAG, &core_idle);
    mpv_get_property(app.mpv, "time-pos", MPV_FORMAT_DOUBLE, &tpos);
    mpv_get_property(app.mpv, "duration", MPV_FORMAT_DOUBLE, &dur);
    mpv_get_property(app.mpv, "width", MPV_FORMAT_INT64, &w);
    mpv_get_property(app.mpv, "height", MPV_FORMAT_INT64, &h);
    int e_vo = mpv_get_property(app.mpv, "current-vo", MPV_FORMAT_STRING, &vo);
    // "no" means software decoding; otherwise the active hwdec (nvdec, ...).
    mpv_get_property(app.mpv, "hwdec-current", MPV_FORMAT_STRING, &hw);
    fprintf(stderr,
            "mpv-vr: mpv core-idle=%d(e=%d) vo=%s(e=%d) hwdec=%s tpos=%.2f "
            "dur=%.2f %lldx%lld\n",
            core_idle, (int)e_ci, vo ? vo : "(none)", (int)e_vo,
            hw ? hw : "(none)", tpos, dur, (long long)w, (long long)h);
    mpv_free(vo);
    mpv_free(hw);
}

static void poll_events(App &app)
{
    XrEventDataBuffer buf = {};
    // Drain the whole event queue. xrPollEvent delivers one event per call and
    // returns XR_EVENT_UNAVAILABLE (a normal result, NOT an error) once the
    // queue is empty — the steady state for most frames. So loop until that,
    // and only die on a real error.
    XrResult r = XR_SUCCESS;
    while (r == XR_SUCCESS) {
        buf = {};
        buf.type = XR_TYPE_EVENT_DATA_BUFFER;
        r = xrPollEvent(app.instance, &buf);
        if (r == XR_EVENT_UNAVAILABLE)
            break; // no more events
        if (r != XR_SUCCESS)
            die("xrPollEvent -> %d (0x%x)", (int)r, (unsigned)r);
        if (buf.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            const auto *e =
                reinterpret_cast<const XrEventDataSessionStateChanged *>(&buf);
            if (e->state == XR_SESSION_STATE_STOPPING ||
                e->state == XR_SESSION_STATE_LOSS_PENDING ||
                e->state == XR_SESSION_STATE_EXITING)
                app.quit = true;
        }
    }
}

static void create_fbos(App &app)
{
    for (size_t i = 0; i < app.swapchains.size(); i++) {
        uint32_t n = 0;
        xrEnumerateSwapchainImages(app.swapchains[i], 0, &n, nullptr);
        app.images[i].resize(n ? n : 1);
        for (auto &img : app.images[i])
            img.type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR;
        xrEnumerateSwapchainImages(app.swapchains[i], (uint32_t)app.images[i].size(),
                                   &n,
                                   reinterpret_cast<XrSwapchainImageBaseHeader *>(
                                       app.images[i].data()));
        app.fbos[i].resize(n);
        fprintf(stderr,
                "mpv-vr: swapchain %zu has %u image(s) ctx-current=%d\n", i, n,
                (int)(eglGetCurrentContext() == app.egl_context));
        for (uint32_t j = 0; j < n; j++) {
            GLuint tex = app.images[i][j].image;
            GLboolean isTex = glIsTexture(tex);
            GLuint fbo = 0;
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, tex, 0);
            GLenum glErr = glGetError();
            GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE || glErr != GL_NO_ERROR)
                fprintf(stderr,
                        "mpv-vr: FBO %zu/%u tex=%u isTexture=%d glErr=0x%x status=0x%x\n",
                        i, j, (unsigned)tex, (int)isTex, (unsigned)glErr,
                        (unsigned)status);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            app.fbos[i][j] = fbo;
        }
    }
}

static void frame_loop(App &app)
{
    XrFrameWaitInfo fw = {};
    fw.type = XR_TYPE_FRAME_WAIT_INFO;
    XrFrameBeginInfo fb = {};
    fb.type = XR_TYPE_FRAME_BEGIN_INFO;

    std::vector<XrView> views(app.viewCount);
    for (auto &v : views)
        v.type = XR_TYPE_VIEW;

    int frame = 0;
    while (!app.quit) {
        poll_events(app);
        if (app.quit)
            break;

        XrFrameState fs = {};
        fs.type = XR_TYPE_FRAME_STATE;
        XR_CHECK(xrWaitFrame(app.session, &fw, &fs));
        // The frame must be begun and ended on EVERY iteration, even when
        // shouldRender is false: Monado only advances the session from
        // READY to SYNCHRONIZED once a frame has been ended (which is what
        // eventually makes shouldRender true), and the next xrWaitFrame
        // blocks until this frame has been begun.
        XR_CHECK(xrBeginFrame(app.session, &fb));

        if (!fs.shouldRender) {
            XrFrameEndInfo fei = {};
            fei.type = XR_TYPE_FRAME_END_INFO;
            fei.displayTime = fs.predictedDisplayTime;
            fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            fei.layerCount = 0;
            fei.layers = nullptr;
            XR_CHECK(xrEndFrame(app.session, &fei));
            continue;
        }

        XrViewState vs = {};
        vs.type = XR_TYPE_VIEW_STATE;
        XrViewLocateInfo vli = {};
        vli.type = XR_TYPE_VIEW_LOCATE_INFO;
        vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        vli.displayTime = fs.predictedDisplayTime;
        vli.space = app.viewSpace;
        uint32_t n = app.viewCount;
        XR_CHECK(xrLocateViews(app.session, &vli, &vs, n, &n, views.data()));

        // xrLocateViews can return an INVALID pose for a frame and still
        // report XR_SUCCESS: e.g. the first rendered frames, before the
        // driver has produced a head pose for this frame's predicted display
        // time. Submitting that zero pose to xrEndFrame is what triggers
        // XR_ERROR_POSE_INVALID ("not a valid quat").
        //
        // As in the working openxr-simple-playground, check the view state
        // and, when the orientation is not valid, end the frame with no
        // layers (skip this frame) instead of submitting a bad pose. The
        // quad appears as soon as tracking delivers a valid pose.
        bool pose_valid =
            (vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) != 0;
        for (uint32_t i = 0; i < n; i++) {
            // Defensive: a driver may set the valid bit with a zero quat.
            const XrQuaternionf *o = &views[i].pose.orientation;
            if (o->x == 0 && o->y == 0 && o->z == 0 && o->w == 0)
                pose_valid = false;
        }
        if (!pose_valid) {
            if (app.skipped < 8 || (app.skipped % 256) == 0) {
                fprintf(stderr,
                    "mpv-vr: no valid head pose (viewStateFlags=0x%x); "
                    "skipping layer [skipped=%ld]. quat={%f %f %f %f} "
                    "pos={%f %f %f}\n",
                    (unsigned)vs.viewStateFlags, app.skipped,
                    views[0].pose.orientation.x, views[0].pose.orientation.y,
                    views[0].pose.orientation.z, views[0].pose.orientation.w,
                    views[0].pose.position.x, views[0].pose.position.y,
                    views[0].pose.position.z);
            }
            app.skipped++;
            XrFrameEndInfo fei = {};
            fei.type = XR_TYPE_FRAME_END_INFO;
            fei.displayTime = fs.predictedDisplayTime;
            fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            fei.layerCount = 0;
            fei.layers = nullptr;
            XR_CHECK(xrEndFrame(app.session, &fei));
            continue;
        }

        if (app.submitted == 0)
            fprintf(stderr,
                "mpv-vr: first valid head pose after %ld skipped frame(s); "
                "submitting layer. quat={%f %f %f %f} pos={%f %f %f}\n",
                app.skipped,
                views[0].pose.orientation.x, views[0].pose.orientation.y,
                views[0].pose.orientation.z, views[0].pose.orientation.w,
                views[0].pose.position.x, views[0].pose.position.y,
                views[0].pose.position.z);
        app.submitted++;

        // Render this frame into the quad's swapchain image. Both eyes see the
        // same image; the quad layer (a real 3D object) provides the stereo.
        {
            // Monado requires acquire BEFORE wait: the acquire reserves an
            // image index, and the wait blocks until the compositor is done
            // displaying that image. (Wait-without-acquire fails with
            // XR_ERROR_CALL_ORDER_INVALID "No image acquired".)
            XrSwapchainImageAcquireInfo ai = {};
            ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO;
            uint32_t idx = 0;
            XR_CHECK(xrAcquireSwapchainImage(app.swapchains[0], &ai, &idx));

            XrSwapchainImageWaitInfo wi = {};
            wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO;
            wi.timeout = INT64_MAX;
            XR_CHECK(xrWaitSwapchainImage(app.swapchains[0], &wi));

            if (app.solid) {
                // Fill the whole swapchain texture with flat turquoise to
                // prove the XR/swapchain/FBO path works, independent of mpv.
                glBindFramebuffer(GL_FRAMEBUFFER, app.fbos[0][idx]);
                glClearColor(0.0f, 1.0f, 1.0f, 1.0f); // turquoise
                glClear(GL_COLOR_BUFFER_BIT);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            } else {
                // Let mpv process pending state (new decoded frame, option
                // changes, ...). We always render below, whether or not a new
                // video frame arrived: the swapchain rotates through several
                // images, so skipping the render on a "no new frame" tick
                // would present a stale/uninitialised image. mpv simply
                // redraws the current frame in that case.
                uint64_t flags = mpv_render_context_update(app.render);
                (void)flags; // MPV_RENDER_UPDATE_FRAME set = new video frame

                mpv_opengl_fbo fbo = {
                    .fbo = (int)app.fbos[0][idx],
                    .w = (int)app.width,
                    .h = (int)app.height,
                    // The real swapchain format, so mpv doesn't guess (a
                    // wrong guess gives washed-out or too-dark colors).
                    .internal_format = (int)app.format,
                };
                // OpenXR GL swapchain textures are top-down relative to mpv's
                // default FBO orientation, so flip vertically.
                int flip_y = 1;
                mpv_render_param rp[] = {
                    { MPV_RENDER_PARAM_OPENGL_FBO, &fbo },
                    { MPV_RENDER_PARAM_FLIP_Y, &flip_y },
                    { MPV_RENDER_PARAM_INVALID, nullptr },
                };
                mpv_render_context_render(app.render, rp);
                // Make sure mpv's GL commands are submitted before the
                // compositor reads the texture.
                glFlush();
            }

            XrSwapchainImageReleaseInfo ri = {};
            ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO;
            XR_CHECK(xrReleaseSwapchainImage(app.swapchains[0], &ri));
            if (app.flat)
                mpv_render_context_report_swap(app.render);
        }

        if (++frame % 60 == 0)
            print_mpv_status(app);

        // One QUAD layer: a single image on a flat screen in front of the
        // viewer. A quad is ONE 3D object with ONE pose (the screen's position)
        // and a size, and both eyes see the same screen (eyeVisibility=BOTH),
        // so each eye views it from its own angle -> correct stereo, no double
        // vision. (A projection layer instead carries one image AND one pose
        // PER EYE and is meant for compositing a per-eye 3D-scene render; fed a
        // flat 2D image it shows the screen once per eye, i.e. twice.)
        float quad_w = 2.0f; // screen width in metres
        float quad_h = quad_w * (float)app.height / (float)app.width; // image aspect
        XrCompositionLayerQuad quad = {};
        quad.type = XR_TYPE_COMPOSITION_LAYER_QUAD;
        quad.space = app.viewSpace;
        quad.pose.orientation = { 0, 0, 0, 1 };
        quad.pose.position = { 0, 0, 0 }; // at viewSpace origin (4 m ahead)
        quad.size = { quad_w, quad_h };
        quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        quad.subImage.swapchain = app.swapchains[0];
        quad.subImage.imageArrayIndex = 0;
        quad.subImage.imageRect.offset = { 0, 0 };
        quad.subImage.imageRect.extent = { (int)app.width, (int)app.height };

        const XrCompositionLayerBaseHeader *layers[1] = {
            reinterpret_cast<const XrCompositionLayerBaseHeader *>(&quad) };
        XrFrameEndInfo fei = {};
        fei.type = XR_TYPE_FRAME_END_INFO;
        fei.displayTime = fs.predictedDisplayTime;
        fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        fei.layerCount = 1;
        fei.layers = layers;
        XR_CHECK(xrEndFrame(app.session, &fei));
    }
}

static void shutdown(App &app)
{
    if (app.render)
        mpv_render_context_free(app.render);
    if (app.mpv)
        mpv_destroy(app.mpv);

    if (app.session) {
        xrEndSession(app.session);
        for (auto s : app.swapchains)
            xrDestroySwapchain(s);
        xrDestroySession(app.session);
    }
    if (app.instance)
        xrDestroyInstance(app.instance);

    for (auto &fbos : app.fbos)
        for (auto fbo : fbos)
            glDeleteFramebuffers(1, &fbo);

    // The EGL context must stay current for the GL deletes above; drop it last.
    if (app.egl_display) {
        eglMakeCurrent(app.egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT);
        if (app.egl_context)
            eglDestroyContext(app.egl_display, app.egl_context);
        eglTerminate(app.egl_display);
    }

    if (app.wl)
        wl_display_disconnect(app.wl);
}

int main(int argc, char **argv)
{
    // usage: mpv-vr [--solid | --flat] [--hwdec=<value>] <video file>
    //   --solid  diagnostic: clear the swapchain to flat turquoise, no mpv
    //             (confirms the XR/swapchain/FBO/composition path, no video)
    //   --flat   render <video file> with mpv, flat on the quad (the real 2D
    //             path); also the default when a video is given without a mode
    //   --hwdec= mpv hwdec value (default "nvdec,nvdec-copy"; "no" = software)
    App app;
    const char *video = nullptr;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--solid") == 0)
            app.solid = true;
        else if (std::strcmp(argv[i], "--flat") == 0)
            app.flat = true;
        else if (std::strncmp(argv[i], "--hwdec=", 8) == 0)
            app.hwdec = argv[i] + 8;
        else if (video == nullptr)
            video = argv[i];
        else
            die("unexpected argument: %s", argv[i]);
    }
    if (app.solid && app.flat)
        die("--solid and --flat are mutually exclusive");
    if (!app.solid && !app.flat) {
        // No explicit mode: a video means --flat; nothing at all is a usage error.
        if (video == nullptr) {
            fprintf(stderr,
                    "usage: mpv-vr --solid <video>   (diagnostic: flat color, no mpv)\n"
                    "       mpv-vr --flat  <video>   (render <video> with mpv on a quad)\n");
            return 1;
        }
        app.flat = true;
    }
    if (app.flat && video == nullptr) {
        fprintf(stderr, "usage: mpv-vr --flat <video file>\n");
        return 1;
    }
    if (app.solid)
        fprintf(stderr, "mpv-vr: --solid mode (flat turquoise, mpv disabled)\n");
    if (app.flat)
        fprintf(stderr, "mpv-vr: --flat mode (rendering '%s' with mpv)\n", video);


    // Wayland first: the EGL display is built off this wl_display.
    app.wl = wl_display_connect(nullptr);
    if (!app.wl)
        die("no WAYLAND_DISPLAY: this player requires Wayland");

    // Headless EGL/GL context (no window/surface): we only render into OpenXR
    // swapchain textures, and we hand this context to Monado.
    app.egl_display = eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND_KHR, app.wl, nullptr);
    if (app.egl_display == EGL_NO_DISPLAY)
        die("eglGetPlatformDisplay(Wayland) failed");
    if (!eglInitialize(app.egl_display, nullptr, nullptr))
        die("eglInitialize failed");
    {
        const EGLint config_attrs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_NONE,
        };
        EGLint num = 0;
        if (!eglChooseConfig(app.egl_display, config_attrs, &app.egl_config, 1, &num) ||
            num == 0)
            die("eglChooseConfig: no desktop-OpenGL config");
        // No profile mask: requesting a core profile is rejected on this
        // driver (EGL_BAD_ATTRIBUTE); a plain desktop-GL 3.0 context works and
        // is all mpv's render API needs.
        const EGLint ctx_attrs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0,
            EGL_NONE,
        };
        app.egl_context =
            eglCreateContext(app.egl_display, app.egl_config, nullptr, ctx_attrs);
        if (!app.egl_context)
            die("eglCreateContext failed");
    }
    if (!eglMakeCurrent(app.egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                        app.egl_context))
        die("eglMakeCurrent failed");

    // OpenXR instance / system / session.
    XrApplicationInfo appInfo = {};
    snprintf(appInfo.applicationName, sizeof appInfo.applicationName, "mpv-vr");
    appInfo.applicationVersion = 1;
    snprintf(appInfo.engineName, sizeof appInfo.engineName, "mpv-vr");
    appInfo.engineVersion = 1;
    appInfo.apiVersion = XR_CURRENT_API_VERSION;

    // XR_MNDX_egl_enable: Monado's EGL context binding.
    // XR_KHR_opengl_enable: OpenGL graphics API; Monado's EGL binding exposes
    // swapchain images as GL textures and requires the
    // xrGetOpenGLGraphicsRequirementsKHR call below.
    const char *instance_exts[] = {
        XR_MNDX_EGL_ENABLE_EXTENSION_NAME,
        XR_KHR_OPENGL_ENABLE_EXTENSION_NAME,
    };
    XrInstanceCreateInfo ic = {};
    ic.type = XR_TYPE_INSTANCE_CREATE_INFO;
    ic.applicationInfo = appInfo;
    ic.enabledExtensionCount =
        (uint32_t)(sizeof instance_exts / sizeof instance_exts[0]);
    ic.enabledExtensionNames = instance_exts;
    XR_CHECK(xrCreateInstance(&ic, &app.instance));

    XrSystemGetInfo sg = {};
    sg.type = XR_TYPE_SYSTEM_GET_INFO;
    sg.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XR_CHECK(xrGetSystem(app.instance, &sg, &app.systemId));

    // Monado's EGL binding is built on the OpenGL graphics API, so its GL
    // requirements must be queried before the session is created. Extension
    // entry points are not linked directly; fetch the pointer first.
    PFN_xrVoidFunction glReqAddr = nullptr;
    XR_CHECK(xrGetInstanceProcAddr(app.instance,
                                   "xrGetOpenGLGraphicsRequirementsKHR",
                                   &glReqAddr));
    if (!glReqAddr)
        die("failed to resolve xrGetOpenGLGraphicsRequirementsKHR");
    auto getGlReq =
        reinterpret_cast<PFN_xrGetOpenGLGraphicsRequirementsKHR>(glReqAddr);
    XrGraphicsRequirementsOpenGLKHR glreq = {};
    glreq.type = XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR;
    XR_CHECK(getGlReq(app.instance, app.systemId, &glreq));
    fprintf(stderr,
            "mpv-vr: runtime supports OpenGL %u.%u .. %u.%u\n",
            (unsigned)XR_VERSION_MAJOR(glreq.minApiVersionSupported),
            (unsigned)XR_VERSION_MINOR(glreq.minApiVersionSupported),
            (unsigned)XR_VERSION_MAJOR(glreq.maxApiVersionSupported),
            (unsigned)XR_VERSION_MINOR(glreq.maxApiVersionSupported));

    // Monado's EGL binding: hand it our EGL display/config/context plus a way
    // to fetch GL entry points. It creates the swapchain textures in our
    // context and reports them back as GL texture names.
    XrGraphicsBindingEGLMNDX binding = {};
    binding.type = XR_TYPE_GRAPHICS_BINDING_EGL_MNDX;
    binding.getProcAddress =
        reinterpret_cast<PFN_xrEglGetProcAddressMNDX>(&eglGetProcAddress);
    binding.display = app.egl_display;
    binding.config = app.egl_config;
    binding.context = app.egl_context;

    XrSessionCreateInfo sc = {};
    sc.type = XR_TYPE_SESSION_CREATE_INFO;
    sc.next = &binding;
    sc.systemId = app.systemId;
    XR_CHECK(xrCreateSession(app.instance, &sc, &app.session));

    // Swapchain size from the runtime's recommendation.
    uint32_t n = 0;
    xrEnumerateViewConfigurationViews(app.instance, app.systemId,
                                      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                      0, &n, nullptr);
    app.viewCount = n;
    if (app.viewCount == 0)
        die("no views in primary stereo configuration");
    std::vector<XrViewConfigurationView> conf(app.viewCount);
    for (auto &v : conf)
        v.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
    xrEnumerateViewConfigurationViews(app.instance, app.systemId,
                                      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                      app.viewCount, &n, conf.data());
    app.viewCount = n;
    app.width = conf[0].recommendedImageRectWidth ? conf[0].recommendedImageRectWidth
                                                  : 1280;
    app.height = conf[0].recommendedImageRectHeight ? conf[0].recommendedImageRectHeight
                                                    : 720;

    // Pick a swapchain format the runtime offers. Prefer plain GL_RGBA8:
    // video pixels are already gamma-encoded, so we want them stored as-is
    // rather than run through sRGB conversion. Fall back to whatever the
    // runtime lists first (0 if it doesn't say).
    uint32_t nf = 0;
    xrEnumerateSwapchainFormats(app.session, 0, &nf, nullptr);
    int64_t format = 0;
    if (nf > 0) {
        std::vector<int64_t> formats(nf);
        xrEnumerateSwapchainFormats(app.session, nf, &nf, formats.data());
        fprintf(stderr, "mpv-vr: runtime swapchain formats:");
        for (auto f : formats)
            fprintf(stderr, " 0x%llx", (unsigned long long)f);
        fputc('\n', stderr);
        format = formats[0];
        for (auto f : formats) {
            if (f == (int64_t)GL_RGBA8) {
                format = f;
                break;
            }
        }
    }
    app.format = format;
    fprintf(stderr, "mpv-vr: using swapchain format 0x%llx%s\n",
            (unsigned long long)format,
            format == (int64_t)GL_RGBA8 ? " (GL_RGBA8)"
            : format == (int64_t)GL_SRGB8_ALPHA8 ? " (GL_SRGB8_ALPHA8)" : "");

    // One swapchain for the quad layer: both eyes see the SAME image, so a
    // single (non per-eye) swapchain is all we need.
    app.swapchains.resize(1);
    {
        XrSwapchainCreateInfo c = {};
        c.type = XR_TYPE_SWAPCHAIN_CREATE_INFO;
        c.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        c.format = format;
        c.sampleCount = 1;
        c.width = app.width;
        c.height = app.height;
        c.faceCount = 1;
        c.arraySize = 1;
        c.mipCount = 1;
        XR_CHECK(xrCreateSwapchain(app.session, &c, &app.swapchains[0]));
    }
    app.images.resize(1);
    app.fbos.resize(1);
    create_fbos(app);

    // Reference space: origin at the quad, QUAD_Z in front of the initial
    // head pose (so the quad is where you're looking when you put the headset on).
    XrReferenceSpaceCreateInfo rs = {};
    rs.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    rs.poseInReferenceSpace.orientation = { 0, 0, 0, 1 };
    rs.poseInReferenceSpace.position = { 0, 0, QUAD_Z };
    XR_CHECK(xrCreateReferenceSpace(app.session, &rs, &app.viewSpace));

    XrSessionBeginInfo sb = {};
    sb.type = XR_TYPE_SESSION_BEGIN_INFO;
    sb.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    XR_CHECK(xrBeginSession(app.session, &sb));

    // mpv: render API into our GL context, then load the video. Only in
    // --flat mode (the --solid diagnostic doesn't use mpv).
    if (app.flat) {
        app.mpv = mpv_create();
        if (!app.mpv)
            die("mpv_create() failed");

        // Options must be set BEFORE mpv_initialize().
        //
        // vo=libmpv: without this mpv opens its own gpu-next window on the
        // desktop and the render context never receives any video (black
        // frames in the FBO). libmpv is the VO that feeds the render API.
        mpv_set_option_string(app.mpv, "vo", "libmpv");
        // There is no real display for mpv to sync to, so sync to audio
        // instead of the default display-sync logic.
        mpv_set_option_string(app.mpv, "video-sync", "audio");
        // Hardware decoding on the NVIDIA GPU. nvdec = zero-copy via
        // CUDA-GL interop; nvdec-copy = same decoder, frames copied back to
        // system memory (more compatible, slower at 8K).
        mpv_set_option_string(app.mpv, "hwdec", app.hwdec.c_str());
        // Allow hwdec for every codec (older mpv builds restrict the list).
        mpv_set_option_string(app.mpv, "hwdec-codecs", "all");

        // mpv_create() only returns a PRE-initialized handle; the core must be
        // explicitly started before it will process commands such as loadfile.
        // Without this, mpv_command() returns MPV_ERROR_UNINITIALIZED (-3) and
        // no video ever loads (black screen, 0x0).
        int einit = mpv_initialize(app.mpv);
        if (einit != 0)
            die("mpv_initialize() -> %d", einit);
        mpv_opengl_init_params glp = { .get_proc_address = gl_get_proc };
        mpv_render_param cp[] = {
            { MPV_RENDER_PARAM_API_TYPE, (void *)MPV_RENDER_API_TYPE_OPENGL },
            { MPV_RENDER_PARAM_OPENGL_INIT_PARAMS, &glp },
            { MPV_RENDER_PARAM_INVALID, nullptr },
        };
        if (mpv_render_context_create(&app.render, app.mpv, cp) < 0)
            die("mpv_render_context_create() failed");
        const char *load[] = { "loadfile", video, "replace", nullptr };
        int e = mpv_command(app.mpv, load);
        fprintf(stderr, "mpv-vr: mpv loadfile('%s') -> %d\n", video, (int)e);
    }

    frame_loop(app);
    shutdown(app);
    return 0;
}
