"""
Neon Rainbow Laser Hands
=========================
Tracks both hands in real time with MediaPipe, and draws an animated
neon-rainbow "laser beam" (plus glowing spark particles) between the two
index fingertips. Everything except the ML inference is rendered with raw
OpenGL (via GLFW + PyOpenGL) as a shader-lit overlay on the live webcam feed.

Controls
--------
  ESC / Q   quit
  B         thicker beam
  N         thinner beam
  P         toggle particle sparks on/off

Requirements
------------
  pip install opencv-python mediapipe glfw PyOpenGL PyOpenGL_accelerate numpy

On first run the script auto-downloads the MediaPipe hand-landmark model
(~8 MB, one-time) into the same folder as this file.
"""

import os
import sys
import time
import math
import ctypes
import colorsys
import urllib.request

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import glfw
from OpenGL.GL import *


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
REQUESTED_CAM_WIDTH, REQUESTED_CAM_HEIGHT = 1280, 720

MAX_PARTICLES = 700
PARTICLES_PER_FRAME = 14
PARTICLE_LIFETIME = 0.6            # seconds
PARTICLE_BASE_SIZE = 20.0          # px, at spawn (shrinks as it dies)

BEAM_HALF_WIDTH = 0.045            # NDC units, i.e. the -1..1 screen space
RAINBOW_SPEED = 0.4                # rainbow cycles per second

INDEX_FINGER_TIP = 8               # MediaPipe hand landmark index

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def ensure_model() -> str:
    """Download the MediaPipe hand-landmark model bundle if it isn't
    already sitting next to this script. Returns the local path."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH
    print("Hand-tracking model not found locally - downloading (~8 MB, one-time)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model saved to:", MODEL_PATH)
    except Exception as exc:
        print(f"Auto-download failed ({exc}).", file=sys.stderr)
        print(
            "Please download it manually from:\n  "
            f"{MODEL_URL}\n"
            "and place it next to this script as 'hand_landmarker.task'.",
            file=sys.stderr,
        )
        sys.exit(1)
    return MODEL_PATH


def px_to_ndc(x_norm: float, y_norm: float) -> tuple:
    """MediaPipe landmark (0..1, origin top-left) -> OpenGL NDC (-1..1, y-up)."""
    return x_norm * 2.0 - 1.0, 1.0 - y_norm * 2.0


def hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


# ----------------------------------------------------------------------
# Shader source
# ----------------------------------------------------------------------
BG_VERT = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTex;
out vec2 vTex;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTex = aTex;
}
"""

BG_FRAG = """
#version 330 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTexture;
void main() {
    FragColor = vec4(texture(uTexture, vTex).rgb, 1.0);
}
"""

BEAM_VERT = """
#version 330 core
layout (location = 0) in vec2 aPos;   // NDC position
layout (location = 1) in vec2 aUV;    // u: 0..1 along beam, v: -1..1 across beam
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

BEAM_FRAG = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform float uTime;
uniform float uSpeed;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    float dist = abs(vUV.y);                          // 0 at center, 1 at edge
    float core = exp(-pow(dist * 7.0, 2.0));           // tight bright core
    float halo = exp(-pow(dist * 2.2, 2.0)) * 0.55;    // soft wide glow
    float glow = clamp(core + halo, 0.0, 1.0);

    float hue = fract(vUV.x - uTime * uSpeed);
    vec3 rainbow = hsv2rgb(vec3(hue, 0.85, 1.0));
    vec3 color = mix(rainbow, vec3(1.0), core * 0.6);  // hot white plasma core

    FragColor = vec4(color, glow);
}
"""

PARTICLE_VERT = """
#version 330 core
layout (location = 0) in vec2 aPos;     // NDC position
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;   // point size in pixels
layout (location = 3) in float aAlpha;
out vec3 vColor;
out float vAlpha;
void main() {
    vColor = aColor;
    vAlpha = aAlpha;
    gl_Position = vec4(aPos, 0.0, 1.0);
    gl_PointSize = aSize;
}
"""

PARTICLE_FRAG = """
#version 330 core
in vec3 vColor;
in float vAlpha;
out vec4 FragColor;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float dist = length(d) * 2.0;              // 0 center .. 1 edge
    float glow = exp(-pow(dist * 2.4, 2.0));
    if (glow < 0.02) discard;
    FragColor = vec4(vColor, glow * vAlpha);
}
"""


def compile_shader(source: str, shader_type) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader


def create_program(vert_src: str, frag_src: str) -> int:
    vs = compile_shader(vert_src, GL_VERTEX_SHADER)
    fs = compile_shader(frag_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


# ----------------------------------------------------------------------
# Background: draws the live camera frame as a full-screen textured quad
# ----------------------------------------------------------------------
class BackgroundRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.program = create_program(BG_VERT, BG_FRAG)
        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "uTexture"), 0)

        # x, y (NDC corners)      u, v (texcoords, v flipped for row-major images)
        verts = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        stride = 4 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # allocate storage once; every frame we only SUBSTITUTE pixels (cheaper)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

    def update_frame(self, rgb_frame: np.ndarray) -> None:
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                         GL_RGB, GL_UNSIGNED_BYTE, rgb_frame)

    def draw(self) -> None:
        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)


# ----------------------------------------------------------------------
# Beam: the neon rainbow line between the two index fingertips
# ----------------------------------------------------------------------
class BeamRenderer:
    def __init__(self):
        self.program = create_program(BEAM_VERT, BEAM_FRAG)
        self.time_loc = glGetUniformLocation(self.program, "uTime")
        speed_loc = glGetUniformLocation(self.program, "uSpeed")
        glUseProgram(self.program)
        glUniform1f(speed_loc, RAINBOW_SPEED)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # 4 vertices * (x, y, u, v) floats, re-uploaded every frame
        glBufferData(GL_ARRAY_BUFFER, 4 * 4 * 4, None, GL_DYNAMIC_DRAW)
        stride = 4 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        self._visible = False

    def update_geometry(self, p0: np.ndarray, p1: np.ndarray, half_width: float) -> None:
        """Build a thin quad stretching from fingertip p0 to fingertip p1."""
        d = p1 - p0
        length = float(math.hypot(d[0], d[1]))
        if length < 1e-5:
            self._visible = False
            return
        nx, ny = -d[1] / length, d[0] / length  # unit vector perpendicular to the beam

        verts = np.array([
            p0[0] - nx * half_width, p0[1] - ny * half_width, 0.0, -1.0,
            p0[0] + nx * half_width, p0[1] + ny * half_width, 0.0,  1.0,
            p1[0] - nx * half_width, p1[1] - ny * half_width, 1.0, -1.0,
            p1[0] + nx * half_width, p1[1] + ny * half_width, 1.0,  1.0,
        ], dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        self._visible = True

    def draw(self, t: float) -> None:
        if not self._visible:
            return
        glUseProgram(self.program)
        glUniform1f(self.time_loc, t)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)


# ----------------------------------------------------------------------
# Particles: glowing sparks scattered along the beam, rendered as
# additive point sprites (numpy arrays on the CPU, streamed to one VBO)
# ----------------------------------------------------------------------
class ParticleSystem:
    def __init__(self, max_particles: int = MAX_PARTICLES):
        self.max_particles = max_particles
        self.pos = np.zeros((0, 2), dtype=np.float32)
        self.vel = np.zeros((0, 2), dtype=np.float32)
        self.color = np.zeros((0, 3), dtype=np.float32)
        self.life = np.zeros((0,), dtype=np.float32)   # 1.0 (born) -> 0.0 (dead)
        self.size = np.zeros((0,), dtype=np.float32)

        self.program = create_program(PARTICLE_VERT, PARTICLE_FRAG)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        stride = 7 * 4  # 2 pos + 3 color + 1 size + 1 alpha, all float32
        glBufferData(GL_ARRAY_BUFFER, self.max_particles * stride, None, GL_STREAM_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(3)
        glBindVertexArray(0)

    def _append(self, pos, vel, color, life, size) -> None:
        self.pos = np.vstack([self.pos, pos])
        self.vel = np.vstack([self.vel, vel])
        self.color = np.vstack([self.color, color])
        self.life = np.concatenate([self.life, life])
        self.size = np.concatenate([self.size, size])
        overflow = len(self.life) - self.max_particles
        if overflow > 0:
            self.pos = self.pos[overflow:]
            self.vel = self.vel[overflow:]
            self.color = self.color[overflow:]
            self.life = self.life[overflow:]
            self.size = self.size[overflow:]

    def spawn_along(self, p0: np.ndarray, p1: np.ndarray, count: int, t: float) -> None:
        """Scatter `count` sparks randomly along the segment p0 -> p1."""
        if count <= 0:
            return
        u = np.random.uniform(0.0, 1.0, size=count).astype(np.float32)
        base = p0[None, :] + (p1 - p0)[None, :] * u[:, None]

        d = p1 - p0
        length = float(math.hypot(d[0], d[1])) + 1e-6
        perp = np.array([-d[1], d[0]], dtype=np.float32) / length
        jitter = np.random.uniform(-0.02, 0.02, size=count).astype(np.float32)
        new_pos = base + perp[None, :] * jitter[:, None]

        angle = np.random.uniform(0.0, 2 * np.pi, size=count)
        speed = np.random.uniform(0.05, 0.25, size=count)
        new_vel = np.stack(
            [np.cos(angle) * speed, np.sin(angle) * speed + 0.05], axis=1
        ).astype(np.float32)

        hues = (u - t * RAINBOW_SPEED) % 1.0
        new_color = np.array(
            [hsv_to_rgb(float(h), 0.85, 1.0) for h in hues], dtype=np.float32
        )

        new_life = np.ones(count, dtype=np.float32)
        new_size = np.random.uniform(
            PARTICLE_BASE_SIZE * 0.6, PARTICLE_BASE_SIZE, size=count
        ).astype(np.float32)

        self._append(new_pos, new_vel, new_color, new_life, new_size)

    def spawn_burst_at(self, point, count: int, t: float, speed: float = 0.05) -> None:
        """Small glow burst anchored at a fingertip (marks where the beam emits from)."""
        if count <= 0:
            return
        angle = np.random.uniform(0.0, 2 * np.pi, size=count)
        spd = np.random.uniform(0.01, speed, size=count)
        new_vel = np.stack([np.cos(angle) * spd, np.sin(angle) * spd], axis=1).astype(np.float32)
        new_pos = np.tile(np.asarray(point, dtype=np.float32), (count, 1))
        base_hue = (t * RAINBOW_SPEED) % 1.0
        new_color = np.array(
            [hsv_to_rgb(base_hue + i * 0.12, 0.6, 1.0) for i in range(count)],
            dtype=np.float32,
        )
        new_life = np.ones(count, dtype=np.float32)
        new_size = np.full(count, PARTICLE_BASE_SIZE * 1.4, dtype=np.float32)
        self._append(new_pos, new_vel, new_color, new_life, new_size)

    def update(self, dt: float) -> None:
        if len(self.life) == 0:
            return
        self.pos += self.vel * dt
        self.vel[:, 1] -= 0.4 * dt          # gentle "gravity" so sparks arc and fall
        self.life -= dt / PARTICLE_LIFETIME
        alive = self.life > 0.0
        self.pos = self.pos[alive]
        self.vel = self.vel[alive]
        self.color = self.color[alive]
        self.life = self.life[alive]
        self.size = self.size[alive]

    def draw(self) -> None:
        n = len(self.life)
        if n == 0:
            return
        alpha = np.clip(self.life, 0.0, 1.0)
        sizes = self.size * alpha  # shrink as they die
        data = np.hstack(
            [self.pos, self.color, sizes[:, None], alpha[:, None]]
        ).astype(np.float32)

        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glDrawArrays(GL_POINTS, 0, n)
        glBindVertexArray(0)


# ----------------------------------------------------------------------
# Hand tracking (MediaPipe Tasks API)
# ----------------------------------------------------------------------
class HandTracker:
    """Wraps MediaPipe's HandLandmarker and returns index-fingertip
    positions, already converted to OpenGL NDC space, for each hand."""

    def __init__(self, model_path: str, max_hands: int = 2,
                 det_conf: float = 0.6, track_conf: float = 0.6):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()
        self._last_timestamp_ms = -1

    def get_index_fingertips(self, rgb_frame: np.ndarray) -> list:
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        tips = []
        for hand_landmarks in result.hand_landmarks:
            tip = hand_landmarks[INDEX_FINGER_TIP]
            tips.append(px_to_ndc(tip.x, tip.y))
        return tips

    def close(self) -> None:
        self.landmarker.close()


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    glfw.set_error_callback(
        lambda code, desc: print(f"GLFW error [{code}]: {desc}", file=sys.stderr)
    )
    if not glfw.init():
        print("Failed to initialize GLFW.", file=sys.stderr)
        sys.exit(1)

    # Open the webcam FIRST so we know the actual frame size the driver
    # gives us (it may silently ignore our requested resolution).
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0).", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_CAM_HEIGHT)

    ok, first_frame = cap.read()
    if not ok:
        print("Could not read a frame from the webcam.", file=sys.stderr)
        cap.release()
        sys.exit(1)
    cam_h, cam_w = first_frame.shape[:2]
    print(f"Camera resolution: {cam_w}x{cam_h}")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.RESIZABLE, False)

    window = glfw.create_window(cam_w, cam_h, "Neon Rainbow Laser Hands", None, None)
    if not window:
        print("Failed to create GLFW window.", file=sys.stderr)
        glfw.terminate()
        cap.release()
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync

    fb_w, fb_h = glfw.get_framebuffer_size(window)
    glViewport(0, 0, fb_w, fb_h)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_PROGRAM_POINT_SIZE)

    background = BackgroundRenderer(cam_w, cam_h)
    beam = BeamRenderer()
    particles = ParticleSystem()

    model_path = ensure_model()
    tracker = HandTracker(model_path)

    state = {"beam_half_width": BEAM_HALF_WIDTH, "particles_on": True}

    def key_callback(win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_B:
            state["beam_half_width"] = min(state["beam_half_width"] * 1.2, 0.25)
        elif key == glfw.KEY_N:
            state["beam_half_width"] = max(state["beam_half_width"] / 1.2, 0.01)
        elif key == glfw.KEY_P:
            state["particles_on"] = not state["particles_on"]

    glfw.set_key_callback(window, key_callback)
    glfw.set_framebuffer_size_callback(
        window, lambda win, w, h: glViewport(0, 0, w, h)
    )

    start_time = time.time()
    prev_time = start_time

    try:
        while not glfw.window_should_close(window):
            now = time.time()
            dt = max(now - prev_time, 1e-4)
            prev_time = now
            t = now - start_time

            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_bgr = cv2.flip(frame_bgr, 1)  # mirror -> feels natural to move in
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            tips = tracker.get_index_fingertips(frame_rgb)
            particles.update(dt)

            beam_visible = len(tips) >= 2
            if beam_visible:
                p0 = np.array(tips[0], dtype=np.float32)
                p1 = np.array(tips[1], dtype=np.float32)
                pulse = 1.0 + 0.15 * math.sin(t * 8.0)
                beam.update_geometry(p0, p1, state["beam_half_width"] * pulse)
                if state["particles_on"]:
                    particles.spawn_along(p0, p1, PARTICLES_PER_FRAME, t)
                    particles.spawn_burst_at(p0, 2, t)
                    particles.spawn_burst_at(p1, 2, t)

            hud = (f"{1.0 / dt:4.0f} FPS | hands: {len(tips)} | "
                   f"B/N width  P particles  Q quit")
            cv2.putText(frame_rgb, hud, (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

            glClear(GL_COLOR_BUFFER_BIT)

            glDisable(GL_BLEND)
            background.update_frame(frame_rgb)
            background.draw()

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # additive glow
            if beam_visible:
                beam.draw(t)
            particles.draw()

            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        cap.release()
        tracker.close()
        glfw.terminate()


if __name__ == "__main__":
    main()
