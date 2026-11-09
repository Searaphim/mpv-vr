// mpv-vr: minimal VR video player for flat (2D/3D) and VR180/VR360 video.
//
// mpv renders into ONE OpenXR swapchain texture via its OpenGL render API (no
// VO window). What the headset shows depends on the mode:
//
//   --flat (default)  the video on a flat quad screen ~4 m in front of the
//                     viewer. Plain 2D, or stereo 3D when a layout is given
//                     (--sbs / --tb): one quad per eye, each showing its own
//                     half of the frame ("3D theater").
//   --vr              equirectangular VR video (VR180, VR360, or anything in
//                     between, set with --hfov / --vfov) wrapped around the
//                     viewer with XR_KHR_composition_layer_equirect2. Needs
//                     --sbs or --tb: one layer per eye.
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
const GLubyte *glGetStringi(GLenum name, GLuint index);
}

#include <mpv/client.h>
#include <mpv/render.h>
#include <mpv/render_gl.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <csignal>
#include <dlfcn.h>
#include <termios.h>
#include <unistd.h>

#ifndef GL_SRGB8_ALPHA8
#define GL_SRGB8_ALPHA8 0x8C43
#endif
#ifndef GL_NUM_EXTENSIONS
#define GL_NUM_EXTENSIONS 0x821D
#endif
#ifndef GL_OUT_OF_MEMORY
#define GL_OUT_OF_MEMORY 0x0505
#endif

#define QUAD_Z (-4.0f)      // flat screen sits this far in front of the head
#define QUAD_WIDTH (2.0f)   // flat screen width in metres

// Keyboard adjustment increments (easy to tune).
#define KEY_ZOOM_STEP       0.05
#define KEY_ALIGNMENT_STEP  0.1   // degrees per arrow-key press
#define KEY_ASPECT_STEP     0.01  // per period/comma press

// Swapchain sizing is measured at runtime (see create_swapchain): the headset's
// pixel density from its reported field of view, the video's own resolution,
// the GL texture limits, the free GPU memory, and a retry loop in case the
// runtime still refuses the size. Only two policy constants are left:
// how much of the currently free GPU memory the swapchain may use (the rest is
// for mpv's decoded frames and scaling), and how much smaller each retry is.
#define SWAPCHAIN_VRAM_SHARE 0.5
#define SWAPCHAIN_SHRINK_STEP 0.75

// Equirect layer radius. Zero means an infinitely distant sphere, which is
// what you want for VR video: only the viewing direction matters, so there is
// no parallax from head movement.
#define VR_SPHERE_RADIUS 0.0f

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

// How the left-eye and right-eye pictures are packed into the video frame.
enum class Layout {
    None,        // one picture (plain 2D video)
    SideBySide,  // left eye = left half, right eye = right half
    TopBottom,   // left eye = top half, right eye = bottom half
};

// Whether each eye's picture is squeezed to fit the frame.
//   Half: squeezed (a 1920x1080 SBS frame holds two 960x1080 pictures)
//   Full: not squeezed (a 3840x1080 SBS frame holds two 1920x1080 pictures)
//   Auto: decide from the frame's shape (see update_eye_aspect)
enum class Packing { Auto, Half, Full };

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
    // Head-locked space, offset QUAD_Z in front of the head: the flat screen
    // lives here, and xrLocateViews is relative to it.
    XrSpace viewSpace{};
    // World-fixed space for VR content (--vr only), so the picture stays put
    // when you turn your head.
    XrSpace localSpace{};
    uint32_t viewCount = 0;
    uint32_t width = 0, height = 0; // swapchain size (BOTH eyes if packed)
    // GL internal format of the swapchain textures (e.g. GL_RGBA8). Passed to
    // mpv as the FBO's internal_format so it doesn't have to guess.
    int64_t format = 0;
    // The runtime's recommended per-eye size (the floor for our own sizing).
    uint32_t rec_w = 0, rec_h = 0;
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
    // Extra mpv options from --mpv=name=value (repeatable), applied after our
    // own settings so they can override e.g. hwdec or video-sync.
    std::vector<std::pair<std::string, std::string>> mpv_opts;

    bool quit = false;

    // ---- Modes and options (set by argument parsing) ----
    // --solid: diagnostic; clear the swapchain to flat turquoise, no mpv
    //   (confirms the XR/swapchain/FBO/composition path on its own).
    bool solid = false;
    // --vr: equirect sphere section around the viewer. Otherwise (the
    // default) the video goes on a flat quad screen.
    bool vr = false;
    bool vr_extension_enabled = false; // runtime supports equirect2 layers
    Layout layout = Layout::None; // --sbs / --tb
    Packing packing = Packing::Auto; // --sbs=half|full, --tb=half|full
    bool swap_eyes = false; // --swap-eyes
    // Alignment of the two eyes' pictures, in degrees (stereo layouts only).
    // halign > 0 moves the pictures apart (left one left, right one right, by
    // half each), so the scene looks farther away; < 0 moves them together.
    // valign > 0 raises the right eye's picture and lowers the left's (half
    // each); < 0 does the opposite. Applied per display eye, whatever
    // --swap-eyes says about which half of the file goes where.
    double halign_deg = 0.0; // --halignment
    double valign_deg = 0.0; // --valignment
    double hfov_deg = 180.0; // --hfov (VR only)
    double vfov_deg = 0.0;   // --vfov (VR only); 0 = derive from the picture
    // --zoom (VR only): enlarges (>1) or shrinks (<1) the picture by scaling
    // the angles it covers. The sphere section can't exceed 360 x 180 degrees,
    // so beyond that the picture is cropped around its centre instead.
    double zoom = 1.0;
    // Aspect ratio (w/h) of ONE eye's picture. Set by --aspect, otherwise
    // derived from the video's size once mpv has loaded it. 0 = not known yet.
    double eye_aspect = 0.0;

    // Log VR angles once.
    bool vr_logged = false;
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

static bool runtime_has_extension(const char *name)
{
    uint32_t n = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &n, nullptr);
    std::vector<XrExtensionProperties> props(n);
    for (auto &p : props) {
        p = {};
        p.type = XR_TYPE_EXTENSION_PROPERTIES;
    }
    xrEnumerateInstanceExtensionProperties(nullptr, n, &n, props.data());
    for (auto &p : props)
        if (std::strcmp(p.extensionName, name) == 0)
            return true;
    return false;
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

// Wrap each swapchain image in an FBO. Returns false if any FBO is incomplete
// or the GL driver ran out of memory allocating the images, so the caller can
// retry with a smaller swapchain.
static bool create_fbos(App &app)
{
    bool all_ok = true;
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
            if (status != GL_FRAMEBUFFER_COMPLETE || glErr == GL_OUT_OF_MEMORY)
                all_ok = false;
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            app.fbos[i][j] = fbo;
        }
    }
    return all_ok;
}

// ---- Swapchain sizing: everything below is measured, not assumed ----

// Query a GL integer; 0 if the driver doesn't support the query.
static GLint gl_get_int(GLenum pname)
{
    GLint v = 0;
    while (glGetError() != GL_NO_ERROR) {}
    glGetIntegerv(pname, &v);
    if (glGetError() != GL_NO_ERROR)
        return 0;
    return v;
}

static bool gl_has_extension(const char *name)
{
    const GLint n = gl_get_int(GL_NUM_EXTENSIONS);
    for (GLint i = 0; i < n; i++) {
        const GLubyte *e = glGetStringi(GL_EXTENSIONS, (GLuint)i);
        if (e && std::strcmp(reinterpret_cast<const char *>(e), name) == 0)
            return true;
    }
    return false;
}

// Free GPU memory in bytes right now, or 0 if the driver can't tell us
// (NVIDIA and AMD can; the swapchain then relies on the retry loop alone).
static uint64_t gl_free_vram_bytes()
{
    if (gl_has_extension("GL_NVX_gpu_memory_info")) {
        // GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, in KiB
        const GLint kib = gl_get_int(0x9049);
        if (kib > 0)
            return (uint64_t)kib * 1024;
    }
    if (gl_has_extension("GL_ATI_meminfo")) {
        // TEXTURE_FREE_MEMORY_ATI: four values, the first is the free pool in KiB
        GLint v[4] = { 0, 0, 0, 0 };
        while (glGetError() != GL_NO_ERROR) {}
        glGetIntegerv(0x87FC, v);
        if (glGetError() == GL_NO_ERROR && v[0] > 0)
            return (uint64_t)v[0] * 1024;
    }
    return 0;
}

// Largest width/height we can render into: the smaller of the GL texture and
// viewport limits. 0 if the driver reports neither.
static uint32_t gl_max_dim()
{
    GLint dim = gl_get_int(GL_MAX_TEXTURE_SIZE);
    GLint vp[2] = { 0, 0 };
    while (glGetError() != GL_NO_ERROR) {}
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, vp);
    if (glGetError() == GL_NO_ERROR) {
        for (int i = 0; i < 2; i++)
            if (vp[i] > 0 && (dim <= 0 || vp[i] < dim))
                dim = vp[i];
    }
    return dim > 0 ? (uint32_t)dim : 0;
}

// Pixels per degree at the CENTRE of the view that the runtime's recommended
// per-eye size delivers, from the field of view xrLocateViews reports. A
// perspective image has f = size / (tan(right) - tan(left)) pixels per unit
// tangent, and at the centre one unit of tangent is one radian. (The average
// over the whole view would overstate it: pixels per degree grow towards the
// edges of a perspective image, and that is not where you look.)
static void headset_pixel_density(const std::vector<XrView> &views,
                                  uint32_t rec_w, uint32_t rec_h,
                                  double &ppd_x, double &ppd_y)
{
    ppd_x = ppd_y = 0;
    for (const XrView &v : views) {
        const double tx = std::tan(v.fov.angleRight) - std::tan(v.fov.angleLeft);
        const double ty = std::tan(v.fov.angleUp) - std::tan(v.fov.angleDown);
        if (std::isfinite(tx) && std::isfinite(ty) && tx > 0 && ty > 0) {
            ppd_x = std::max(ppd_x, rec_w / tx * M_PI / 180.0);
            ppd_y = std::max(ppd_y, rec_h / ty * M_PI / 180.0);
        }
    }
}

// Vertical coverage of the VR picture: --vfov, or derived from hfov and the
// per-eye aspect (at most 180).
static double vr_vfov(const App &app)
{
    if (app.vfov_deg > 0)
        return app.vfov_deg;
    if (app.eye_aspect <= 0)
        return std::min(app.hfov_deg, 180.0);
    return std::min(180.0, app.hfov_deg / app.eye_aspect);
}

static bool read_video_size(App &app, int64_t &w, int64_t &h)
{
    w = h = 0;
    if (!app.mpv)
        return false;
    return mpv_get_property(app.mpv, "width", MPV_FORMAT_INT64, &w) >= 0 &&
           mpv_get_property(app.mpv, "height", MPV_FORMAT_INT64, &h) >= 0 &&
           w > 0 && h > 0;
}

static void destroy_swapchain(App &app)
{
    for (auto &fbos : app.fbos) {
        for (auto fbo : fbos)
            glDeleteFramebuffers(1, &fbo);
        fbos.clear();
    }
    for (auto s : app.swapchains)
        xrDestroySwapchain(s);
    app.swapchains.clear();
    app.images.clear();
    app.fbos.clear();
}

// Create the one swapchain that holds everything (both eyes packed like the
// video, or a single picture for plain 2D); the layers pick out the part for
// each eye. Done lazily from the frame loop because the headset's field of
// view (xrLocateViews) and the video's size are only known once the session
// is running and the video has loaded.
//
// Per-eye size, starting from the runtime's recommendation (the floor):
//   flat: the recommendation. The screen covers only a fraction of the view,
//         so it is already more than the panel can show of it.
//   --vr: enough texels to match the headset's pixel density across the angle
//         the picture covers (times the zoom), but never more than the video
//         has pixels, and never more than the GL limits allow.
// Then the size is verified and, if needed, shrunk step by step until the
// runtime accepts it, the FBOs are complete, and (where the driver reports
// free GPU memory) the images fit in our share of it.
static void create_swapchain(App &app, const std::vector<XrView> &views,
                             int64_t src_w, int64_t src_h)
{
    const bool sbs = app.layout == Layout::SideBySide;
    const bool tb = app.layout == Layout::TopBottom;
    const uint32_t px = sbs ? 2 : 1; // pictures across
    const uint32_t py = tb ? 2 : 1;  // pictures down

    uint32_t eye_w = app.rec_w, eye_h = app.rec_h;
    double ppd_x = 0, ppd_y = 0;
    if (app.vr) {
        headset_pixel_density(views, app.rec_w, app.rec_h, ppd_x, ppd_y);
        if (ppd_x > 0 && ppd_y > 0) {
            // The zoom enlarges the picture without adding pixels, so the
            // texture grows with it to keep the visible part as sharp.
            double want_w = ppd_x * app.hfov_deg * app.zoom;
            double want_h = ppd_y * vr_vfov(app) * app.zoom;
            // Texels beyond the video's own pixels add nothing.
            if (src_w > 0 && src_h > 0) {
                want_w = std::min(want_w, (double)(sbs ? src_w / 2 : src_w));
                want_h = std::min(want_h, (double)(tb ? src_h / 2 : src_h));
            }
            eye_w = std::max<uint32_t>(app.rec_w, (uint32_t)want_w);
            eye_h = std::max<uint32_t>(app.rec_h, (uint32_t)want_h);
        }
    }

    const uint32_t max_dim = gl_max_dim();
    if (max_dim) {
        eye_w = std::min(eye_w, max_dim / px);
        eye_h = std::min(eye_h, max_dim / py);
    }

    const uint64_t free_vram = gl_free_vram_bytes();
    const uint64_t budget = (uint64_t)((double)free_vram * SWAPCHAIN_VRAM_SHARE);
    // Bytes per pixel of the swapchain format (RGBA8 / sRGB8_A8 = 4; assume
    // the worst, a 16-bit float format, for anything else).
    const uint32_t bpp =
        (app.format == (int64_t)GL_RGBA8 || app.format == (int64_t)GL_SRGB8_ALPHA8) ? 4 : 8;

    for (;;) {
        uint32_t w = (eye_w * px) & ~1u; // even, so the halves split exactly
        uint32_t h = (eye_h * py) & ~1u;
        const bool at_floor = eye_w <= app.rec_w && eye_h <= app.rec_h;

        // Drain stale GL errors so the ones checked below come from the
        // allocation inside xrCreateSwapchain.
        while (glGetError() != GL_NO_ERROR) {}

        XrSwapchainCreateInfo c = {};
        c.type = XR_TYPE_SWAPCHAIN_CREATE_INFO;
        c.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        c.format = app.format;
        c.sampleCount = 1;
        c.width = w;
        c.height = h;
        c.faceCount = 1;
        c.arraySize = 1;
        c.mipCount = 1;
        XrSwapchain sc = XR_NULL_HANDLE;
        const XrResult r = xrCreateSwapchain(app.session, &c, &sc);

        bool ok = (r == XR_SUCCESS);
        if (!ok) {
            fprintf(stderr, "mpv-vr: swapchain %ux%u refused by the runtime "
                            "(XrResult %d)\n", w, h, (int)r);
        } else {
            app.swapchains.assign(1, sc);
            app.width = w;
            app.height = h;
            app.images.assign(1, {});
            app.fbos.assign(1, {});
            ok = create_fbos(app);
            if (!ok)
                fprintf(stderr, "mpv-vr: swapchain %ux%u could not be "
                                "allocated (incomplete FBO or GL out of memory)\n",
                        w, h);
            const uint64_t need =
                (uint64_t)w * h * bpp * (uint64_t)app.fbos[0].size();
            if (ok && budget && need > budget && !at_floor) {
                fprintf(stderr, "mpv-vr: swapchain %ux%u needs %.0f MB, over "
                                "our %.0f MB share of free GPU memory\n",
                        w, h, need / 1048576.0, budget / 1048576.0);
                ok = false;
            }
            if (!ok)
                destroy_swapchain(app);
        }
        if (ok)
            break;

        if (at_floor)
            die("could not create a swapchain even at the runtime's "
                "recommended size %ux%u", app.rec_w, app.rec_h);
        eye_w = std::max(app.rec_w, (uint32_t)(eye_w * SWAPCHAIN_SHRINK_STEP));
        eye_h = std::max(app.rec_h, (uint32_t)(eye_h * SWAPCHAIN_SHRINK_STEP));
    }

    fprintf(stderr, "mpv-vr: swapchain %ux%u, %zu image(s), %.0f MB each "
                    "(runtime recommends %ux%u per eye)\n",
            app.width, app.height, app.fbos[0].size(),
            (double)app.width * app.height * bpp / 1048576.0, app.rec_w,
            app.rec_h);
    if (app.vr) {
        fprintf(stderr, "mpv-vr: sizing inputs: headset %.1f x %.1f px/deg at "
                        "the centre, video %lldx%lld, GL max %u, ",
                ppd_x, ppd_y, (long long)src_w, (long long)src_h, max_dim);
        if (free_vram)
            fprintf(stderr, "free GPU memory %.0f MB\n", free_vram / 1048576.0);
        else
            fprintf(stderr, "free GPU memory unknown\n");
    }
}

// Work out the aspect ratio of ONE eye's picture from the loaded video's size
// (unless --aspect already set it). Called every frame until it succeeds,
// because the video loads asynchronously.
//
// The frame's own aspect ratio is `frame`. How it relates to one eye's
// picture depends on the layout and on whether the eyes are squeezed:
//
//   no layout             eye = frame
//   SBS, full (unsqueezed) eye = frame / 2     (each eye gets half the width)
//   SBS, half (squeezed)   eye = frame         (squeeze undone on display)
//   TB,  full (unsqueezed) eye = frame * 2     (each eye gets half the height)
//   TB,  half (squeezed)   eye = frame
//
// With Packing::Auto in flat mode we guess from the frame's shape. No normal
// movie frame is wider than ~2.4 and no full-SBS frame is narrower than ~2.7,
// so SBS wider than 3.0 is full. For TB a full-TB frame of a normal movie is
// at most ~1.2 and a half-TB frame is at least ~1.33, so TB narrower than
// 1.25 is full (a narrower margin than SBS: use =half/=full if it guesses
// wrong). In VR mode Auto means full: VR video is not normally squeezed.
static void update_eye_aspect(App &app)
{
    if (app.eye_aspect > 0 || !app.mpv)
        return;
    int64_t w = 0, h = 0;
    if (mpv_get_property(app.mpv, "width", MPV_FORMAT_INT64, &w) < 0 ||
        mpv_get_property(app.mpv, "height", MPV_FORMAT_INT64, &h) < 0 ||
        w <= 0 || h <= 0)
        return; // video not loaded yet

    // Prefer the display aspect (accounts for non-square pixels); fall back
    // to the plain pixel ratio.
    double frame = (double)w / (double)h;
    double dar = 0;
    if (mpv_get_property(app.mpv, "video-params/aspect", MPV_FORMAT_DOUBLE,
                        &dar) >= 0 && dar > 0)
        frame = dar;

    const char *what = "single picture";
    bool full = false;
    switch (app.layout) {
    case Layout::SideBySide:
        full = app.packing == Packing::Full ||
               (app.packing == Packing::Auto && (app.vr || frame > 3.0));
        app.eye_aspect = full ? frame / 2.0 : frame;
        what = full ? "full SBS" : "half SBS";
        break;
    case Layout::TopBottom:
        full = app.packing == Packing::Full ||
               (app.packing == Packing::Auto && (app.vr || frame < 1.25));
        app.eye_aspect = full ? frame * 2.0 : frame;
        what = full ? "full TB" : "half TB";
        break;
    case Layout::None:
        app.eye_aspect = frame;
        break;
    }
    fprintf(stderr,
            "mpv-vr: video %lldx%lld (frame aspect %.3f) -> %s, per-eye "
            "aspect %.3f\n",
            (long long)w, (long long)h, frame, what, app.eye_aspect);
}

// The part of the swapchain that holds one eye's picture (eye 0 = left).
// With --swap-eyes the two parts trade places.
static XrSwapchainSubImage eye_sub_image(const App &app, int eye)
{
    XrSwapchainSubImage s = {};
    s.swapchain = app.swapchains[0];
    s.imageArrayIndex = 0;
    const int w = (int)app.width, h = (int)app.height;
    const int src = app.swap_eyes ? 1 - eye : eye;
    switch (app.layout) {
    case Layout::SideBySide:
        s.imageRect.offset = { src * (w / 2), 0 };
        s.imageRect.extent = { w / 2, h };
        break;
    case Layout::TopBottom:
        s.imageRect.offset = { 0, src * (h / 2) };
        s.imageRect.extent = { w, h / 2 };
        break;
    case Layout::None:
        s.imageRect.offset = { 0, 0 };
        s.imageRect.extent = { w, h };
        break;
    }
    return s;
}

// Keep only the central part of a sub-image: fraction fx of its width and fy of
// its height. Used by --vr --zoom when the enlarged picture would cover more
// of the sphere than exists. The equirect mapping is linear in angle, so the
// centre of the rectangle is the centre of the view.
static void crop_sub_image(XrSwapchainSubImage &s, double fx, double fy)
{
    const int w = s.imageRect.extent.width, h = s.imageRect.extent.height;
    const int nw = std::max(1, (int)std::lround(w * fx));
    const int nh = std::max(1, (int)std::lround(h * fy));
    s.imageRect.offset.x += (w - nw) / 2;
    s.imageRect.offset.y += (h - nh) / 2;
    s.imageRect.extent.width = nw;
    s.imageRect.extent.height = nh;
}

// Orientation that turns a layer's picture by a small angle: yaw (positive
// moves the picture to the LEFT) then pitch (positive moves it UP). Used by
// --vr, where the sphere has infinite radius so sliding a layer sideways would
// do nothing; rotating it gives the same visual shift.
static XrQuaternionf alignment_orientation(double yaw_deg, double pitch_deg)
{
    const double d2r = M_PI / 180.0;
    const double sy = std::sin(yaw_deg * d2r * 0.5), cy = std::cos(yaw_deg * d2r * 0.5);
    const double sx = std::sin(pitch_deg * d2r * 0.5), cx = std::cos(pitch_deg * d2r * 0.5);
    // yaw (about +Y) * pitch (about +X)
    XrQuaternionf q;
    q.x = (float)(cy * sx);
    q.y = (float)(sy * cx);
    q.z = (float)(-sy * sx);
    q.w = (float)(cy * cx);
    return q;
}

// ---- Keyboard input from the terminal ----
//
// Keys are read from the terminal mpv-vr was started in (stdin), so THAT
// terminal needs keyboard focus, not the headset. It is put in non-canonical,
// no-echo mode with non-blocking reads (VMIN=0, VTIME=0), so checking it once
// per frame never waits. ISIG stays on, so Ctrl-C still works. The original
// settings are restored on every way out: normal exit, die() (which calls
// exit), and SIGINT/SIGTERM/SIGHUP.
static termios g_tty_saved;
static volatile sig_atomic_t g_tty_raw = 0;

static void terminal_restore()
{
    if (g_tty_raw) {
        g_tty_raw = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &g_tty_saved);
    }
}

static void terminal_signal(int sig)
{
    terminal_restore(); // tcsetattr, signal and raise are async-signal-safe
    signal(sig, SIG_DFL);
    raise(sig);
}

// Returns false (and leaves the terminal alone) if stdin is not a terminal,
// or not in the foreground (changing a background job's terminal would stop
// the process).
static bool terminal_raw_enable()
{
    if (!isatty(STDIN_FILENO) || tcgetpgrp(STDIN_FILENO) != getpgrp())
        return false;
    if (tcgetattr(STDIN_FILENO, &g_tty_saved) != 0)
        return false;
    termios t = g_tty_saved;
    t.c_lflag &= ~(ICANON | ECHO);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &t) != 0)
        return false;
    g_tty_raw = 1;
    atexit(terminal_restore);
    signal(SIGINT, terminal_signal);
    signal(SIGTERM, terminal_signal);
    signal(SIGHUP, terminal_signal);
    return true;
}

static void toggle_pause(App &app)
{
    int paused = 0;
    if (mpv_get_property(app.mpv, "pause", MPV_FORMAT_FLAG, &paused) < 0)
        return;
    paused = !paused;
    mpv_set_property(app.mpv, "pause", MPV_FORMAT_FLAG, &paused);
    fprintf(stderr, "mpv-vr: %s\n", paused ? "paused" : "resumed");
}

// Read whatever keys were typed since the last frame and act on them. Other
// keys (and the bytes of escape sequences, e.g. arrow keys) are ignored.
static void poll_keyboard(App &app)
{
    if (!g_tty_raw || !app.mpv)
        return;

    // Arrow keys arrive as ANSI escape sequences (ESC [ A/B/C/D), and a
    // sequence can be split across consecutive frame polls.
    static int escape_state = 0;
    char buf[64];
    const ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
    for (ssize_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)buf[i];
        if (escape_state == 1) {
            escape_state = (c == '[') ? 2 : 0;
            continue;
        }
        if (escape_state == 2) {
            escape_state = 0;
            switch (c) {
            case 'A': app.valign_deg = std::min(20.0, app.valign_deg + KEY_ALIGNMENT_STEP); break;
            case 'B': app.valign_deg = std::max(-20.0, app.valign_deg - KEY_ALIGNMENT_STEP); break;
            case 'C': app.halign_deg = std::min(20.0, app.halign_deg + KEY_ALIGNMENT_STEP); break;
            case 'D': app.halign_deg = std::max(-20.0, app.halign_deg - KEY_ALIGNMENT_STEP); break;
            default: break;
            }
            fprintf(stderr, "mpv-vr: alignment horizontal=%.2f deg, vertical=%.2f deg\n",
                    app.halign_deg, app.valign_deg);
            continue;
        }
        if (c == 0x1b) {
            escape_state = 1;
            continue;
        }

        switch (c) {
        case ' ':
            toggle_pause(app);
            break;
        case '+':
        case '=': // convenient on layouts where '+' requires Shift
        case '-':
            app.zoom = std::max(0.1, std::min(10.0,
                app.zoom + (c == '-' ? -KEY_ZOOM_STEP : KEY_ZOOM_STEP)));
            fprintf(stderr, "mpv-vr: zoom=%.2f\n", app.zoom);
            if (app.vr) {
                destroy_swapchain(app);
                app.vr_logged = false;
            }
            break;
        case '.':
        case ',':
            update_eye_aspect(app);
            if (app.eye_aspect > 0) {
                app.eye_aspect = std::max(0.05, std::min(20.0,
                    app.eye_aspect + (c == '.' ? KEY_ASPECT_STEP : -KEY_ASPECT_STEP)));
                fprintf(stderr, "mpv-vr: per-eye aspect=%.3f\n", app.eye_aspect);
            } else {
                fprintf(stderr, "mpv-vr: aspect is not available until the video loads\n");
            }
            break;
        case 'v':
        case 'V':
            if (!app.vr_extension_enabled) {
                fprintf(stderr, "mpv-vr: VR mode unavailable: runtime lacks %s\n",
                        XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME);
            } else if (app.layout == Layout::None) {
                fprintf(stderr, "mpv-vr: VR mode needs --sbs or --tb stereo layout\n");
            } else if (!app.vr) {
                app.vr = true;
                app.vr_logged = false;
                destroy_swapchain(app);
                fprintf(stderr, "mpv-vr: switched to VR mode\n");
            }
            break;
        case 'f':
        case 'F':
            if (app.vr) {
                app.vr = false;
                destroy_swapchain(app);
                fprintf(stderr, "mpv-vr: switched to flat mode\n");
            }
            break;
        default:
            break;
        }
    }
}

static void end_frame_empty(App &app, XrTime displayTime)
{
    XrFrameEndInfo fei = {};
    fei.type = XR_TYPE_FRAME_END_INFO;
    fei.displayTime = displayTime;
    fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    fei.layerCount = 0;
    fei.layers = nullptr;
    XR_CHECK(xrEndFrame(app.session, &fei));
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
        poll_keyboard(app);
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
            end_frame_empty(app, fs.predictedDisplayTime);
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
        // picture appears as soon as tracking delivers a valid pose.
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
            end_frame_empty(app, fs.predictedDisplayTime);
            continue;
        }

        // Learn the per-eye aspect from the video once it has loaded.
        update_eye_aspect(app);

        // VR with no explicit --vfov needs the aspect to know how tall the
        // sphere section is. Until the video size is known there is nothing
        // sensible to show, so submit no layers for these first frames.
        if (app.vr && app.vfov_deg <= 0 && app.eye_aspect <= 0) {
            end_frame_empty(app, fs.predictedDisplayTime);
            continue;
        }

        // Create the swapchain now that the headset's field of view is known
        // (the pose is valid) and, for --vr, the video has loaded: its size
        // caps how many texels are worth having. Until the video is there
        // --vr has nothing to show, so it submits no layers.
        if (app.swapchains.empty()) {
            int64_t src_w = 0, src_h = 0;
            const bool have_size = read_video_size(app, src_w, src_h);
            if (app.vr && !have_size) {
                end_frame_empty(app, fs.predictedDisplayTime);
                continue;
            }
            create_swapchain(app, views, src_w, src_h);
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

        // Render this frame into the swapchain image. For a packed layout the
        // whole side-by-side / top-bottom frame is stretched over the whole
        // texture (keepaspect=no), so each eye's half of the video lands
        // exactly on its half of the texture.
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
            if (!app.solid)
                mpv_render_context_report_swap(app.render);
        }

        if (++frame % 60 == 0)
            print_mpv_status(app);

        // Build the layers. A stereo layout gets TWO layers at the same pose,
        // each visible to one eye only and showing its own part of the
        // swapchain; no layout gets ONE layer visible to both eyes.
        const bool stereo = app.layout != Layout::None;
        const uint32_t layer_count = stereo ? 2 : 1;
        XrCompositionLayerQuad quads[2] = {};
        XrCompositionLayerEquirect2KHR eqs[2] = {};
        const XrCompositionLayerBaseHeader *layers[2] = {};

        // Flat: a quad is ONE 3D object with ONE pose (the screen's position)
        // and a size, so each eye views it from its own angle -> correct
        // stereo, no double vision. (A projection layer instead carries one
        // image AND one pose PER EYE and is meant for per-eye 3D-scene
        // renders; fed a flat 2D image it would show the screen twice.)
        // The screen's shape is the picture's: one eye's aspect ratio (the
        // diagnostic --solid has no video, so it uses the swapchain's shape).
        double aspect = app.eye_aspect > 0
                            ? app.eye_aspect
                            : (app.solid ? (double)app.width / (double)app.height
                                         : 16.0 / 9.0);
        const float quad_h = (float)(QUAD_WIDTH / aspect);

        // VR: how much of the sphere the picture covers (hfov x vfov), then
        // the zoom applied on top. Zoom scales both angles; the sphere section
        // can't exceed 360 x 180 degrees, so whatever would stick out is
        // cropped from the picture's edges (crop_x / crop_y = the fraction of
        // the picture that stays visible), keeping the magnification exact.
        double vfov_deg = 0;
        double show_h = 0, show_v = 0; // angles the layer actually spans
        double crop_x = 1.0, crop_y = 1.0;
        if (app.vr) {
            vfov_deg = vr_vfov(app);
            const double want_h = app.hfov_deg * app.zoom;
            const double want_v = vfov_deg * app.zoom;
            show_h = std::min(want_h, 360.0);
            show_v = std::min(want_v, 180.0);
            crop_x = show_h / want_h;
            crop_y = show_v / want_v;
            if (!app.vr_logged) {
                fprintf(stderr,
                        "mpv-vr: VR sphere section: hfov=%.1f vfov=%.1f deg%s\n",
                        app.hfov_deg, vfov_deg,
                        app.vfov_deg > 0 ? "" : " (vfov derived from aspect)");
                if (app.zoom != 1.0)
                    fprintf(stderr,
                            "mpv-vr: zoom %.2f -> showing %.1f x %.1f deg, "
                            "%.0f%% x %.0f%% of the picture\n",
                            app.zoom, show_h, show_v, crop_x * 100.0,
                            crop_y * 100.0);
                app.vr_logged = true;
            }
        }

        for (uint32_t i = 0; i < layer_count; i++) {
            const XrEyeVisibility vis =
                !stereo ? XR_EYE_VISIBILITY_BOTH
                        : (i == 0 ? XR_EYE_VISIBILITY_LEFT
                                  : XR_EYE_VISIBILITY_RIGHT);
            const XrSwapchainSubImage sub = eye_sub_image(app, (int)i);

            // Alignment: each eye's picture moves by half of the setting, the
            // left eye one way and the right eye the other.
            const double side = stereo ? (i == 0 ? -0.5 : 0.5) : 0.0;
            const double shift_right_deg = side * app.halign_deg;
            const double shift_up_deg = side * app.valign_deg;

            if (app.vr) {
                // Equirect2: the compositor maps the picture onto a section
                // of a sphere centred on the viewer. The picture's centre
                // faces forward in the layer's space; the section spans
                // hfov horizontally and +-vfov/2 vertically.
                XrCompositionLayerEquirect2KHR &e = eqs[i];
                e.type = XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR;
                e.space = app.localSpace;
                e.eyeVisibility = vis;
                e.subImage = sub;
                crop_sub_image(e.subImage, crop_x, crop_y);
                e.pose.orientation =
                    alignment_orientation(-shift_right_deg, shift_up_deg);
                e.pose.position = { 0, 0, 0 };
                e.radius = VR_SPHERE_RADIUS;
                e.centralHorizontalAngle = (float)(show_h * M_PI / 180.0);
                e.upperVerticalAngle = (float)(show_v * 0.5 * M_PI / 180.0);
                e.lowerVerticalAngle = -(float)(show_v * 0.5 * M_PI / 180.0);
                layers[i] = reinterpret_cast<const XrCompositionLayerBaseHeader *>(&e);
            } else {
                XrCompositionLayerQuad &q = quads[i];
                q.type = XR_TYPE_COMPOSITION_LAYER_QUAD;
                q.space = app.viewSpace;
                q.pose.orientation = { 0, 0, 0, 1 };
                // At the viewSpace origin (4 m ahead), slid sideways/up so
                // the picture shifts by the alignment angle as seen from the
                // head (QUAD_Z is negative, hence the minus).
                const double d2r = M_PI / 180.0;
                q.pose.position = {
                    (float)(-QUAD_Z * std::tan(shift_right_deg * d2r)),
                    (float)(-QUAD_Z * std::tan(shift_up_deg * d2r)),
                    0.0f };
                q.size = { QUAD_WIDTH, quad_h };
                q.eyeVisibility = vis;
                q.subImage = sub;
                layers[i] = reinterpret_cast<const XrCompositionLayerBaseHeader *>(&q);
            }
        }

        XrFrameEndInfo fei = {};
        fei.type = XR_TYPE_FRAME_END_INFO;
        fei.displayTime = fs.predictedDisplayTime;
        fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        fei.layerCount = layer_count;
        fei.layers = layers;
        XR_CHECK(xrEndFrame(app.session, &fei));
    }
}

static void shutdown(App &app)
{
    terminal_restore();
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

// ---- Argument parsing helpers ----

// Match "--name=value" or "--name value". Returns the value, or nullptr if
// argv[i] is not this flag. Advances i when the value is the next argument.
static const char *flag_value(int argc, char **argv, int &i, const char *name)
{
    size_t n = std::strlen(name);
    if (std::strncmp(argv[i], name, n) != 0)
        return nullptr;
    if (argv[i][n] == '=')
        return argv[i] + n + 1;
    if (argv[i][n] == '\0') {
        if (i + 1 >= argc)
            die("%s needs a value", name);
        return argv[++i];
    }
    return nullptr; // a different flag that merely starts with this name
}

static double parse_number(const char *s, const char *flag, double lo, double hi)
{
    char *end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !std::isfinite(v))
        die("%s: '%s' is not a number", flag, s);
    if (v < lo || v > hi)
        die("%s: %g is out of range (%g .. %g)", flag, v, lo, hi);
    return v;
}

// Match "--sbs", "--sbs=half" or "--sbs=full" (same for --tb).
static bool match_layout_flag(const char *arg, const char *name, Packing &p)
{
    size_t n = std::strlen(name);
    if (std::strncmp(arg, name, n) != 0)
        return false;
    if (arg[n] == '\0') {
        p = Packing::Auto;
        return true;
    }
    if (arg[n] != '=')
        return false;
    const char *v = arg + n + 1;
    if (std::strcmp(v, "half") == 0)
        p = Packing::Half;
    else if (std::strcmp(v, "full") == 0)
        p = Packing::Full;
    else
        die("%s expects =half or =full, got '%s'", name, v);
    return true;
}

static void print_usage()
{
    fprintf(stderr,
        "usage: mpv-vr [options] <video file>\n"
        "\n"
        "mode (default --flat):\n"
        "  --flat              video on a flat screen in front of you\n"
        "  --vr                equirectangular VR video wrapped around you\n"
        "                      (needs --sbs or --tb)\n"
        "  --solid             diagnostic: flat turquoise, no mpv, no video needed\n"
        "\n"
        "stereo layout (none = plain 2D video):\n"
        "  --sbs[=half|full]   side-by-side: left eye = left half\n"
        "  --tb[=half|full]    top-bottom:   left eye = top half\n"
        "                      half = each eye squeezed to half the frame,\n"
        "                      full = each eye unsqueezed. Without =half/=full:\n"
        "                      guessed from the frame shape in --flat, full in --vr\n"
        "  --swap-eyes         the file has the right eye first\n"
        "  --halignment=<deg>  horizontal alignment of the two pictures, -20..20.\n"
        "                      positive moves them apart (scene looks farther\n"
        "                      away), negative moves them together. Try 0.1-1\n"
        "  --valignment=<deg>  vertical alignment, -20..20. positive raises the\n"
        "                      right picture and lowers the left, negative the\n"
        "                      reverse\n"
        "  --aspect=<x.xx>     aspect ratio (w/h) of ONE eye's picture; overrides\n"
        "                      the value worked out from the video\n"
        "\n"
        "VR angles (--vr only):\n"
        "  --hfov=<deg>        horizontal coverage, 1..360 (default 180)\n"
        "  --vfov=<deg>        vertical coverage, 1..180 (default: derived from\n"
        "                      hfov and the per-eye aspect, at most 180)\n"
        "  --zoom=<factor>     enlarge (>1) or shrink (<1) the picture, 0.1..10\n"
        "                      (default 1). The picture can't cover more than\n"
        "                      360 x 180 deg, so past that its edges are cropped\n"
        "\n"
        "other:\n"
        "  --hwdec=<value>     mpv hwdec (default nvdec,nvdec-copy; no = software)\n"
        "  --mpv=<name>[=<value>]\n"
        "                      pass any mpv option; repeat for several, e.g.\n"
        "                      --mpv=scale=ewa_lanczos --mpv=\"sub-file=/my dir/a.srt\"\n"
        "                      (no value = yes, no-<name> = no). Applied after\n"
        "                      mpv-vr's own settings; vo and keepaspect are fixed\n"
        "\n"
        "keys (type in the terminal mpv-vr runs in, not in the headset):\n"
        "  space               pause / resume\n"
        "  numpad + / -        zoom in / out (0.05 per press; VR mode)\n"
        "  up / down           vertical eye alignment +/- 0.1 deg\n"
        "  left / right        horizontal eye alignment +/- 0.1 deg\n"
        "  . / ,               per-eye aspect +/- 0.01\n"
        "  v / f               switch to VR / flat mode\n");
}

int main(int argc, char **argv)
{
    App app;
    const char *video = nullptr;
    bool flat_flag = false;
    bool have_hfov = false, have_vfov = false, have_zoom = false;
    Layout layout_flag = Layout::None;
    for (int i = 1; i < argc; i++) {
        const char *v = nullptr;
        Packing p = Packing::Auto;
        if (std::strcmp(argv[i], "--solid") == 0)
            app.solid = true;
        else if (std::strcmp(argv[i], "--flat") == 0)
            flat_flag = true;
        else if (std::strcmp(argv[i], "--vr") == 0)
            app.vr = true;
        else if (std::strcmp(argv[i], "--swap-eyes") == 0)
            app.swap_eyes = true;
        else if (match_layout_flag(argv[i], "--sbs", p) ||
                 match_layout_flag(argv[i], "--tb", p)) {
            Layout l = std::strncmp(argv[i], "--sbs", 5) == 0
                           ? Layout::SideBySide
                           : Layout::TopBottom;
            if (layout_flag != Layout::None)
                die("--sbs and --tb are mutually exclusive");
            layout_flag = l;
            app.packing = p;
        } else if ((v = flag_value(argc, argv, i, "--hfov"))) {
            app.hfov_deg = parse_number(v, "--hfov", 1.0, 360.0);
            have_hfov = true;
        } else if ((v = flag_value(argc, argv, i, "--vfov"))) {
            app.vfov_deg = parse_number(v, "--vfov", 1.0, 180.0);
            have_vfov = true;
        } else if ((v = flag_value(argc, argv, i, "--zoom"))) {
            app.zoom = parse_number(v, "--zoom", 0.1, 10.0);
            have_zoom = true;
        } else if ((v = flag_value(argc, argv, i, "--aspect"))) {
            app.eye_aspect = parse_number(v, "--aspect", 0.05, 20.0);
        } else if ((v = flag_value(argc, argv, i, "--halignment"))) {
            app.halign_deg = parse_number(v, "--halignment", -20.0, 20.0);
        } else if ((v = flag_value(argc, argv, i, "--valignment"))) {
            app.valign_deg = parse_number(v, "--valignment", -20.0, 20.0);
        } else if ((v = flag_value(argc, argv, i, "--hwdec"))) {
            app.hwdec = v;
        } else if ((v = flag_value(argc, argv, i, "--mpv"))) {
            // --mpv=name=value. Also accepted, as on mpv's own command line:
            // a leading "--", a bare name meaning "yes" (--mpv=pause), and
            // "no-name" meaning "name=no" (--mpv=no-audio).
            std::string opt = v;
            while (!opt.empty() && opt[0] == '-')
                opt.erase(0, 1);
            std::string name = opt, value = "yes";
            const size_t eq = opt.find('=');
            if (eq != std::string::npos) {
                name = opt.substr(0, eq);
                value = opt.substr(eq + 1);
            } else if (name.size() > 3 && name.compare(0, 3, "no-") == 0) {
                name.erase(0, 3);
                value = "no";
            }
            if (name.empty())
                die("--mpv needs an option, e.g. --mpv=scale=ewa_lanczos");
            // These two are what makes mpv render into our swapchain: without
            // them you get a desktop window or a distorted picture.
            if (name == "vo" || name == "keepaspect")
                die("--mpv=%s: mpv-vr sets this itself and it can't be changed",
                    name.c_str());
            app.mpv_opts.emplace_back(name, value);
        } else if (std::strncmp(argv[i], "--", 2) == 0) {
            print_usage();
            die("unknown option: %s", argv[i]);
        } else if (video == nullptr) {
            video = argv[i];
        } else {
            die("unexpected argument: %s", argv[i]);
        }
    }
    app.layout = layout_flag;

    // Mode checks.
    if (app.solid && (flat_flag || app.vr))
        die("--solid cannot be combined with --flat or --vr");
    if (flat_flag && app.vr)
        die("--flat and --vr are mutually exclusive");
    if (app.solid && app.layout != Layout::None)
        die("--solid cannot be combined with --sbs or --tb");
    if ((have_hfov || have_vfov || have_zoom) && !app.vr)
        die("--hfov, --vfov and --zoom only apply with --vr");
    if (app.vr && app.layout == Layout::None)
        die("--vr needs a stereo layout: add --sbs or --tb "
            "(mono VR video is not supported yet)");
    if (app.swap_eyes && app.layout == Layout::None)
        fprintf(stderr, "mpv-vr: warning: --swap-eyes has no effect without "
                        "--sbs or --tb\n");
    if ((app.halign_deg != 0 || app.valign_deg != 0) &&
        app.layout == Layout::None)
        fprintf(stderr, "mpv-vr: warning: --halignment/--valignment have no "
                        "effect without --sbs or --tb\n");
    if (app.solid && !app.mpv_opts.empty())
        fprintf(stderr, "mpv-vr: warning: --mpv options have no effect with "
                        "--solid (mpv is not used)\n");
    if (!app.solid && video == nullptr) {
        print_usage();
        return 1;
    }

    if (app.solid)
        fprintf(stderr, "mpv-vr: --solid mode (flat turquoise, mpv disabled)\n");
    else
        fprintf(stderr, "mpv-vr: %s mode, layout=%s, rendering '%s' with mpv\n",
                app.vr ? "--vr" : "--flat",
                app.layout == Layout::SideBySide ? "side-by-side"
                : app.layout == Layout::TopBottom ? "top-bottom" : "none",
                video);
    if (app.eye_aspect > 0)
        fprintf(stderr, "mpv-vr: per-eye aspect set by --aspect: %.3f\n",
                app.eye_aspect);
    if (app.halign_deg != 0 || app.valign_deg != 0)
        fprintf(stderr, "mpv-vr: eye alignment: horizontal %.3f deg, vertical "
                        "%.3f deg\n", app.halign_deg, app.valign_deg);

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
    // XR_KHR_composition_layer_equirect2: the sphere-section layer used by
    // --vr only.
    std::vector<const char *> instance_exts = {
        XR_MNDX_EGL_ENABLE_EXTENSION_NAME,
        XR_KHR_OPENGL_ENABLE_EXTENSION_NAME,
    };
    app.vr_extension_enabled =
        runtime_has_extension(XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME);
    if (app.vr && !app.vr_extension_enabled)
        die("the OpenXR runtime does not offer %s, which --vr needs",
            XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME);
    if (app.vr_extension_enabled)
        instance_exts.push_back(XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME);
    XrInstanceCreateInfo ic = {};
    ic.type = XR_TYPE_INSTANCE_CREATE_INFO;
    ic.applicationInfo = appInfo;
    ic.enabledExtensionCount = (uint32_t)instance_exts.size();
    ic.enabledExtensionNames = instance_exts.data();
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
    // The runtime's recommended per-eye size is the floor for our own sizing.
    // The swapchain itself is created later, from the frame loop (see
    // create_swapchain), once the headset's field of view and the video's size
    // are known.
    app.rec_w = conf[0].recommendedImageRectWidth
                    ? conf[0].recommendedImageRectWidth : 1280;
    app.rec_h = conf[0].recommendedImageRectHeight
                    ? conf[0].recommendedImageRectHeight : 720;
    fprintf(stderr, "mpv-vr: runtime recommends %ux%u per eye\n", app.rec_w,
            app.rec_h);

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

    // The swapchain (one for everything: both eyes' pictures are packed into
    // it, or it is a single picture for plain 2D) is created from the frame
    // loop, see create_swapchain.

    // Flat screen space: origin at the quad, QUAD_Z in front of the head (so
    // the quad is where you're looking). Head-locked: it follows the head.
    XrReferenceSpaceCreateInfo rs = {};
    rs.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    rs.poseInReferenceSpace.orientation = { 0, 0, 0, 1 };
    rs.poseInReferenceSpace.position = { 0, 0, QUAD_Z };
    XR_CHECK(xrCreateReferenceSpace(app.session, &rs, &app.viewSpace));

    // VR space: fixed in the world, origin where the headset started, so the
    // picture stays put when you turn your head.
    if (app.vr_extension_enabled) {
        XrReferenceSpaceCreateInfo ls = {};
        ls.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
        ls.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
        ls.poseInReferenceSpace.orientation = { 0, 0, 0, 1 };
        ls.poseInReferenceSpace.position = { 0, 0, 0 };
        XR_CHECK(xrCreateReferenceSpace(app.session, &ls, &app.localSpace));
    }

    XrSessionBeginInfo sb = {};
    sb.type = XR_TYPE_SESSION_BEGIN_INFO;
    sb.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    XR_CHECK(xrBeginSession(app.session, &sb));

    // mpv: render API into our GL context, then load the video. Not in the
    // --solid diagnostic, which doesn't use mpv.
    if (!app.solid) {
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
        // Stretch the whole video frame to fill the swapchain instead of
        // letterboxing it. For packed stereo this makes each eye's half of
        // the video land exactly on its half of the texture; for plain 2D the
        // screen's shape comes from the video (not the texture), so there
        // are no black bars baked into the picture.
        mpv_set_option_string(app.mpv, "keepaspect", "no");
        // Hardware decoding on the NVIDIA GPU. nvdec = zero-copy via
        // CUDA-GL interop; nvdec-copy = same decoder, frames copied back to
        // system memory (more compatible, slower at 8K).
        mpv_set_option_string(app.mpv, "hwdec", app.hwdec.c_str());
        // Allow hwdec for every codec (older mpv builds restrict the list).
        mpv_set_option_string(app.mpv, "hwdec-codecs", "all");

        // User options (--mpv=name=value), applied after ours so they can
        // override e.g. hwdec or video-sync. A rejected option is a typo or
        // something this mpv build doesn't have: say so instead of ignoring it.
        for (const auto &o : app.mpv_opts) {
            const int eopt = mpv_set_option_string(app.mpv, o.first.c_str(),
                                                   o.second.c_str());
            if (eopt < 0)
                die("mpv rejected option '%s=%s': %s", o.first.c_str(),
                    o.second.c_str(), mpv_error_string(eopt));
            fprintf(stderr, "mpv-vr: mpv option %s=%s\n", o.first.c_str(),
                    o.second.c_str());
        }

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

    // Keyboard controls from the terminal (not in --solid, which has no mpv).
    if (!app.solid) {
        if (terminal_raw_enable())
            fprintf(stderr, "mpv-vr: keys (type in this terminal): "
                            "space=pause, numpad +/-=zoom, arrows=alignment, "
                            "./,=aspect, V=VR, F=flat\n");
        else
            fprintf(stderr, "mpv-vr: stdin is not a foreground terminal, "
                            "keyboard controls disabled\n");
    }

    frame_loop(app);
    shutdown(app);
    return 0;
}
