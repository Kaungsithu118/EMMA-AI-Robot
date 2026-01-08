# -*- coding: utf-8 -*-
"""
Emma Robot — EVE UI + Voice + Webcam + Face Emotion + YouTube + Online MP3 + Image Search
Follows your original structure, with:
 • Ollama (llama3.2:3b) for answers
 • Vosk for listening
 • MediaPipe Face Detection + FaceMesh for robust detection & emotion
 • Mirrored camera for both photo & face-scan (toggle via CAMERA_MIRROR)
 • NEW: "please search ... image/photo" command that fetches and shows an image

Phrases (type or speak):
  • "capturing image" / "capture image" / "take my picture" / "capture my photo" / "shot me photo"
  • "scan face" / "scanning face" / "scan file face"
  • "please search <query> photo" / "search <query> image" / "find <query> picture"
  • "open <words> in youtube"
  • "play <wor
  • "play music video <words>" / "play <words> on youtube"
  • "open the music <mp3-url>" / "open music <mp3-url>" / "play mp3 <mp3-url>"
  • "open/go to <site>", "open/go to youtube", "stop music", "what time is it"
"""

import os, io, re, math, json, time, queue, tempfile, threading, webbrowser
from time import sleep
from datetime import datetime

import pygame, pyaudio, vosk
from gtts import gTTS
from cvzone.SerialModule import SerialObject
import cv2, numpy as np, requests

# NEW: robust face detect + landmarks
import mediapipe as mp

# Try yt_dlp (optional for auto-opening the top YouTube result)
try:
    import yt_dlp
    HAVE_YTDLP = True
except Exception:
    HAVE_YTDLP = False

# ---------------- Config ----------------
ARDUINO_DIGITS = 3
HOME_POS = [180, 0, 90]
VOSK_MODEL_PATH = "../Resources/vosk-model-en-us-0.22"

# Answering via local Ollama (no Google for answers)
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"


# ---- Conversational Follow-up Memory ----
FOLLOWUP_CUES = ("tell me more", "explain more", "continue", "go on", "more details", "why", "how")
SYSTEM_RULES = (
    "You are EMMA, a friendly, concise AI. Reply in under 80 words unless asked for detail. "
    "If the user says 'tell me more' or similar, continue on the same topic with deeper detail. "
    "When topics involve current public figures or fresh news, clarify date context."
)

class ConversationManager:
    def __init__(self, max_turns:int=12):
        self.max_turns = max_turns
        self.messages = [{"role":"system","content": SYSTEM_RULES}]
        self.last_summary = None
        self.last_title = None

    def _is_followup(self, text:str)->bool:
        t = (text or "").strip().lower()
        if t in FOLLOWUP_CUES:
            return True
        if len(t.split()) <= 4 and any(k in t for k in FOLLOWUP_CUES):
            return True
        return False

    def on_user(self, text:str):
        msg = text.strip()
        if self._is_followup(msg) and self.last_summary:
            msg = (f"(Follow-up on: {self.last_title or 'previous topic'}). "
                   f"Use prior context: {self.last_summary}\n\nUser: {msg}")
        self.messages.append({"role":"user","content": msg})
        self._trim()

    def on_assistant(self, text:str):
        self.messages.append({"role":"assistant","content": text})
        self._trim()
        # Update running summary: simple compact first 700 chars
        t = (text or "").strip()
        if t:
            title = t.split(".")[0][:90].strip()
            self.last_title = title or "Previous topic"
            self.last_summary = t[:700]

    def payload(self):
        sys = [self.messages[0]]
        rest = self.messages[1:][-self.max_turns*2:]
        return sys + rest

    def _trim(self):
        sys = [self.messages[0]]
        rest = self.messages[1:]
        if len(rest) > self.max_turns*2:
            rest = rest[-self.max_turns*2:]
        self.messages = sys + rest

CM = ConversationManager(max_turns=14)

UI_WIDTH, UI_HEIGHT = 1220, 740
UI_FPS = 60
HEADER_H = 70
PAD = 16
SIDEBAR_W = 480
INPUT_H = 56

CHAT_MAX_MESSAGES = 220
BUBBLE_MAX_CHARS = 96
FONT_NAME = None
FONT_SIZE = 18
TITLE_FONT_SIZE = 26

SOUNDS = {"listen": "../Resources/listen.mp3", "convert": "../Resources/convert.mp3"}

# Make capture & scan show the same mirrored view as a front camera
CAMERA_MIRROR = True

# ---------------- State ----------------
shutdown_flag = False
speaking_flag = False
presentation_active = False
mic_enabled = True

chat_lock = threading.Lock()
# message payload: str OR {"image": Surface, "caption": str}
chat_messages = []
status_text = ""

position_lock = threading.Lock()
last_positions = HOME_POS[:]
arduino = None

input_text = ""
user_text_queue = queue.Queue()

# scroll state
chat_scroll_px = 0
chat_content_height = 0
auto_scroll_to_bottom = True
scroll_dragging = False
scroll_drag_dy = 0
_last_chat_rect = None
_last_scroll_track = None
_last_scroll_thumb = None

# pygame handles
screen = None
clock = None
font = None
font_title = None
sfx_listen = None
sfx_convert = None


# ---------------- Chat helpers ----------------
def add_message(role, text):
    global auto_scroll_to_bottom
    with chat_lock:
        t = (text or "").strip()
        if not t:
            return
        chat_messages.append((role, t))
        if len(chat_messages) > CHAT_MAX_MESSAGES:
            del chat_messages[:-CHAT_MAX_MESSAGES]
    if auto_scroll_to_bottom or chat_scroll_px <= 2:
        set_chat_scroll("bottom", None)

def add_image_message(role, surf, caption=""):
    global auto_scroll_to_bottom
    if surf is None:
        return
    max_w = 360
    w, h = surf.get_width(), surf.get_height()
    if w > max_w:
        k = max_w / float(w)
        surf = pygame.transform.smoothscale(surf, (int(w * k), int(h * k)))
    payload = {"image": surf.convert(), "caption": caption or ""}
    with chat_lock:
        chat_messages.append((role, payload))
        if len(chat_messages) > CHAT_MAX_MESSAGES:
            del chat_messages[:-CHAT_MAX_MESSAGES]
    if auto_scroll_to_bottom or chat_scroll_px <= 2:
        set_chat_scroll("bottom", None)

def set_status(text):
    """Small header status note."""
    global status_text
    status_text = text or ""

def wrap_text(text, max_chars=BUBBLE_MAX_CHARS):
    words = (text or "").split()
    lines, line_len, line = [], 0, []
    for w in words:
        extra = len(w) + (1 if line else 0)
        if line_len + extra > max_chars:
            lines.append(" ".join(line))
            line, line_len = [w], len(w)
        else:
            line.append(w)
            line_len += extra
    if line:
        lines.append(" ".join(line))
    return lines or [""]


# ---------------- UI ----------------
def ui_init():
    global screen, clock, font, font_title, sfx_listen, sfx_convert
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.display.init()
    try:
        pygame.mixer.init()
    except Exception:
        pass
    pygame.font.init()
    pygame.event.set_allowed([
        pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEWHEEL,
        pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION
    ])
    screen = pygame.display.set_mode((UI_WIDTH, UI_HEIGHT))
    pygame.display.set_caption("Emma Robot — Soul")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
    font_title = pygame.font.SysFont(FONT_NAME, TITLE_FONT_SIZE, bold=True)
    try:
        if os.path.exists(SOUNDS["listen"]):
            sfx_listen = pygame.mixer.Sound(SOUNDS["listen"])
    except Exception:
        sfx_listen = None
    try:
        if os.path.exists(SOUNDS["convert"]):
            sfx_convert = pygame.mixer.Sound(SOUNDS["convert"])
    except Exception:
        sfx_convert = None

def play_sfx(kind):
    snd = sfx_listen if kind == "listen" else sfx_convert if kind == "convert" else None
    if snd:
        try:
            snd.play()
        except Exception:
            pass

def vgrad(surface, rect, top_rgb, bottom_rgb):
    x, y, w, h = rect
    for i in range(h):
        t = i / float(h)
        c = (
            int(top_rgb[0] * (1 - t) + bottom_rgb[0] * t),
            int(top_rgb[1] * (1 - t) + bottom_rgb[1] * t),
            int(top_rgb[2] * (1 - t) + bottom_rgb[2] * t),
        )
        pygame.draw.line(surface, c, (x, y + i), (x + w, y + i))

def draw_header():
    hdr = pygame.Surface((UI_WIDTH - 2 * PAD, HEADER_H), pygame.SRCALPHA)
    pygame.draw.rect(hdr, (20, 32, 50, 220), hdr.get_rect(), border_radius=14)
    pygame.draw.rect(hdr, (160, 220, 255, 90), hdr.get_rect(), 1, border_radius=14)
    screen.blit(hdr, (PAD, PAD))
    screen.blit(font_title.render("EMMA • AI Robot Soul", True, (236, 246, 255)), (PAD + 20, PAD + 14))
    screen.blit(font.render(f"Mic: {'ON' if mic_enabled else 'OFF'}   ·   F2: toggle mic   ·   Enter: send   ·   Esc: quit",
                            True, (175, 210, 255)), (PAD + 20, PAD + 40))
    if status_text:
        st = font.render(status_text, True, (130, 230, 210))
        screen.blit(st, (UI_WIDTH - PAD - st.get_width() - 8, PAD + 22))


# --------- EVE avatar ----------
def draw_eve(cx, cy, height_px, t):
    H = float(height_px)
    W = H * (320/470)
    scale = H / 470.0
    SS = 2.0
    SW, SH = int(W*SS), int(H*SS)
    surf = pygame.Surface((SW, SH), pygame.SRCALPHA)
    def S(v): return int(v*scale*SS)

    white       = (255,255,255,242)
    rim_white   = (236,244,255,120)
    visor_black = (8,10,12,255)
    visor_rim   = (88,126,170,135)
    eye_glow    = (88,176,255,130)
    highlight   = (255,255,255,56)
    subtle_hi   = (255,255,255,36)

    shadow_w = int(W*1.02)
    shadow_h = int(H*0.085)
    shadow = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
    spread = int(shadow_w * (0.90 + 0.03*math.sin(t*2.0)))
    pygame.draw.ellipse(shadow, (0,0,0,75), (shadow_w//2 - spread//2, 0, spread, shadow_h))
    screen.blit(shadow, (int(cx - shadow_w/2), int(cy + H*0.41)))

    bob  = int(6*SS*scale*math.sin(t*1.6))
    tilt = 2.0*math.sin(t*0.9)

    body = pygame.Surface((SW, SH), pygame.SRCALPHA)
    body_rect = (S(40), S(170 + bob/SS), S(240), S(260))
    pygame.draw.ellipse(body, white, body_rect)
    pygame.draw.ellipse(body, rim_white, body_rect, max(1, S(2)))
    shoulder_mask = pygame.Surface((SW, SH), pygame.SRCALPHA)
    pygame.draw.ellipse(shoulder_mask, (0,0,0,80), (S(36), S(156 + bob/SS), S(248), S(68)))
    body.blit(shoulder_mask, (0,0), special_flags=pygame.BLEND_RGBA_SUB)

    ring_layer = pygame.Surface((SW, SH), pygame.SRCALPHA)
    outer = (S(78), S(214 + bob/SS), S(184), S(54))
    inner = (S(86), S(222 + bob/SS), S(168), S(38))
    pygame.draw.ellipse(ring_layer, (205,220,235,190), outer)
    inner_cut = pygame.Surface((SW, SH), pygame.SRCALPHA)
    pygame.draw.ellipse(inner_cut, (0,0,0,255), inner)
    ring_layer.blit(inner_cut, (0,0), special_flags=pygame.BLEND_RGBA_SUB)
    pygame.draw.ellipse(ring_layer, (70,90,110,60), (S(86), S(246 + bob/SS), S(168), S(16)))
    body.blit(ring_layer, (0,0))
    pygame.draw.ellipse(body, (255,255,255,48), (S(72), S(198 + bob/SS), S(190), S(76)))
    surf.blit(body, (0,0))

    head = pygame.Surface((SW, SH), pygame.SRCALPHA)
    head_rect = (S(70), S(60 + bob/SS), S(180), S(145))
    pygame.draw.ellipse(head, white, head_rect)
    pygame.draw.ellipse(head, highlight, (S(92), S(78 + bob/SS),  S(136), S(54)))
    pygame.draw.ellipse(head, subtle_hi, (S(88), S(108 + bob/SS), S(148), S(62)))
    surf.blit(head, (0,0))

    visor = pygame.Surface((SW, SH), pygame.SRCALPHA)
    visor_rect = (S(86), S(98 + bob/SS), S(148), S(88))
    pygame.draw.ellipse(visor, visor_black, visor_rect)
    pygame.draw.ellipse(visor, visor_rim,   visor_rect, max(1, S(2)))
    pygame.draw.ellipse(visor, (220,232,246,92), (S(104), S(116 + bob/SS), S(36), S(20)))
    pygame.draw.ellipse(visor, (220,232,246,72), (S(178), S(122 + bob/SS), S(28), S(14)))
    surf.blit(visor, (0,0))

    eyes = pygame.Surface((SW, SH), pygame.SRCALPHA)
    ex_l, ex_r = S(126), S(196)
    ey = S(154 + bob/SS)
    breathe = 1.0 + 0.05*math.sin(t*2.6)
    w_core, h_core = S(34*breathe), S(24*breathe)
    sweep = (t*260.0) % S(78)
    def draw_eye(cx, cy):
        pygame.draw.ellipse(eyes, eye_glow, (cx - S(28), cy - S(20), S(56), S(40)))
        for i, alpha in enumerate((235, 205, 170, 120)):
            w = w_core + S(i*4); h = h_core + S(i*3)
            pygame.draw.ellipse(eyes, (120,210,255,alpha), (cx - w//2, cy - h//2, w, h))
        for k in range(-4,5):
            yy = cy + k*S(4)
            band = abs((k*S(4)) - (sweep - S(39))) < S(7)
            a = 160 if band else 90
            pygame.draw.line(eyes, (130,220,255,a), (cx - S(22), yy), (cx + S(22), yy), 1)
    draw_eye(ex_l, ey); draw_eye(ex_r, ey)
    surf.blit(eyes, (0,0))

    arms = pygame.Surface((SW, SH), pygame.SRCALPHA)
    arm_l = (S(36),  S(220 + bob/SS), S(74), S(150))
    arm_r = (S(210), S(220 + bob/SS), S(74), S(150))
    pygame.draw.ellipse(arms, white, arm_l)
    pygame.draw.ellipse(arms, white, arm_r)
    pygame.draw.ellipse(arms, rim_white, arm_l, max(1, S(2)))
    pygame.draw.ellipse(arms, rim_white, arm_r, max(1, S(2)))
    pygame.draw.ellipse(arms, (255,255,255,62), (S(52),  S(246 + bob/SS), S(36), S(22)))
    pygame.draw.ellipse(arms, (255,255,255,62), (S(232), S(246 + bob/SS), S(36), S(22)))
    surf.blit(arms, (0,0))

    surf = pygame.transform.rotozoom(surf, tilt, 1.0)
    final = pygame.transform.smoothscale(surf, (int(W), int(H)))
    screen.blit(final, (int(cx - W/2), int(cy - H/2)))


# ---------------- Chat (images + visible scrollbar) ----------------
def set_chat_scroll(where, chat_rect):
    global chat_scroll_px, auto_scroll_to_bottom
    max_scroll = max(0, chat_content_height - (chat_rect.height if chat_rect else 0) + 20)
    if where == "bottom":
        chat_scroll_px = 0
        auto_scroll_to_bottom = True
    else:
        chat_scroll_px = max(0, min(chat_scroll_px + where, max_scroll))
        auto_scroll_to_bottom = (chat_scroll_px == 0)

def _scrollbar_geometry(chat_rect):
    track_w = 12
    track = pygame.Rect(chat_rect.right - track_w - 6, chat_rect.y + 8, track_w, chat_rect.height - 16)
    view_h = chat_rect.height - 4
    max_scroll = max(0, chat_content_height - view_h)
    th = max(24, int(track.height * (view_h / float(max(chat_content_height, 1)))))
    y_off = int((track.height - th) * (0 if max_scroll == 0 else chat_scroll_px / float(max_scroll)))
    thumb = pygame.Rect(track.x + 2, track.y + (track.height - th - y_off), track.width - 4, th)
    return track, thumb

def _render_bubble(role, payload, max_width):
    if isinstance(payload, dict) and "image" in payload:
        img = payload["image"]; caption = payload.get("caption", "")
        w, h = img.get_width(), img.get_height()
        caps = [font.render(l, True, (12, 16, 20)) for l in wrap_text(caption, 60)] if caption else []
        text_h = sum(s.get_height() for s in caps) + (10 if caps else 0)
        bw = min(max(w + 20, max((s.get_width() for s in caps), default=0) + 28), max_width)
        bh = h + text_h + 20
    else:
        lines = wrap_text(str(payload))
        surfs = [font.render(l, True, (12, 16, 20)) for l in lines]
        bw = min(max((s.get_width() for s in surfs), default=0) + 28, max_width)
        bh = sum(s.get_height() for s in surfs) + 18

    col, rim = ((170, 220, 255, 235), (200, 240, 255, 150)) if role == "you" else \
               ((210, 255, 230, 235), (210, 255, 220, 150)) if role == "emma" else \
               ((235, 235, 235, 220), (255, 255, 255, 120))

    bub = pygame.Surface((bw, bh), pygame.SRCALPHA)
    pygame.draw.rect(bub, col, (0, 0, bw, bh), border_radius=14)
    pygame.draw.rect(bub, rim, (0, 0, bw, bh), 1, border_radius=14)

    if isinstance(payload, dict) and "image" in payload:
        img = payload["image"]; caption = payload.get("caption", "")
        bub.blit(img, ((bw - img.get_width()) // 2, 10))
        ty = 10 + img.get_height() + (6 if caption else 0)
        for s in [font.render(l, True, (12, 16, 20)) for l in wrap_text(caption, 60)]:
            bub.blit(s, ((bw - s.get_width()) // 2, ty)); ty += s.get_height()
    else:
        ty = 9
        for s in [font.render(l, True, (12, 16, 20)) for l in wrap_text(str(payload))]:
            bub.blit(s, (12, ty)); ty += s.get_height()
    return bub, bh

def draw_chat_and_input(x, y, w, h):
    global chat_content_height, _last_chat_rect, _last_scroll_track, _last_scroll_thumb
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    vgrad(panel, (0, 0, w, h), (16, 28, 48), (12, 22, 40))
    pygame.draw.rect(panel, (150, 220, 255, 80), (0, 0, w, h), 1, border_radius=16)
    screen.blit(panel, (x, y))

    chat_rect = pygame.Rect(x + 14, y + 14, w - 28, h - INPUT_H - 22)
    _last_chat_rect = chat_rect
    pygame.draw.rect(screen, (24, 42, 64, 160), chat_rect, border_radius=12)

    with chat_lock:
        yy = 10
        for role, payload in chat_messages:
            _b, bh = _render_bubble(role, payload, max_width=chat_rect.width - 24)
            yy += bh + 10
        chat_content_height = max(yy + 10, chat_rect.height)
        content_w = chat_rect.width - 24
        content = pygame.Surface((content_w, chat_content_height), pygame.SRCALPHA)
        y_cursor = 10
        for role, payload in chat_messages:
            bub, bh = _render_bubble(role, payload, max_width=content_w)
            bx = 10 if role == "you" else content_w - bub.get_width() - 10 if role == "emma" else (content_w - bub.get_width()) // 2
            content.blit(bub, (bx, y_cursor)); y_cursor += bh + 10

    view_h = chat_rect.height - 4
    max_scroll = max(0, chat_content_height - view_h)
    scroll_px = 0 if auto_scroll_to_bottom else max(0, min(chat_scroll_px, max_scroll))
    view_start = max(0, chat_content_height - view_h - scroll_px)
    screen.blit(content, (chat_rect.x + 6, chat_rect.y + 2), area=pygame.Rect(0, view_start, content.get_width(), view_h))

    track, thumb = _scrollbar_geometry(chat_rect)
    _last_scroll_track, _last_scroll_thumb = track, thumb
    pygame.draw.rect(screen, (120, 160, 200, 60), track, border_radius=6)
    pygame.draw.rect(screen, (160, 220, 255, 180), thumb, border_radius=6)
    pygame.draw.rect(screen, (210, 240, 255, 200), thumb, 1, border_radius=6)

    input_rect = pygame.Rect(x + 14, y + h - INPUT_H - 10, w - 28, INPUT_H)
    pygame.draw.rect(screen, (26, 46, 68, 220), input_rect, border_radius=12)
    pygame.draw.rect(screen, (140, 220, 255), input_rect, 1, border_radius=12)
    hint = "Type your question and press Enter…"
    txt = input_text if input_text else hint
    col = (220, 250, 255) if input_text else (155, 185, 205)
    screen.blit(font.render(txt, True, col), (input_rect.x + 12, input_rect.y + 14))


# ---------------- Camera / Media helpers ----------------
def _cv2_to_surface(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
    return surf.convert()

def get_camera_frame(index=0, warmup_frames=12):
    """Try DSHOW then MSMF; warm up frames; require minimal brightness."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for be in backends:
        try:
            cap = cv2.VideoCapture(index, be)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = False, None
            for _ in range(max(4, warmup_frames)):
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.03)
                    continue
                if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 10:
                    break
                time.sleep(0.03)
            cap.release()
            if ok and frame is not None and np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 5:
                return frame
        except Exception:
            try: cap.release()
            except Exception: pass
            continue
    return None

def capture_with_countdown(index=0, seconds=0):
    if seconds > 0:
        for i in range(seconds, 0, -1):
            set_status(f"Photo in {i}…"); time.sleep(1.0)
    set_status("Capturing Image")
    frame = get_camera_frame(index)
    set_status("")
    if frame is None:
        return None, None
    # Mirror to match scan preview
    if CAMERA_MIRROR:
        frame = cv2.flip(frame, 1)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(tempfile.gettempdir(), f"emma_photo_{ts}.jpg")
    cv2.imwrite(out, frame)
    return out, _cv2_to_surface(frame)


# ======= Face Scan using MediaPipe (high accuracy) =======
mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_W   = [61, 291]
MOUTH_V   = [13, 14]
EYE_OUTER = [33, 263]

def _dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def _ear(pts):
    p1,p2,p3,p4,p5,p6 = pts
    return ( _dist(p2,p6) + _dist(p3,p5) ) / (2.0 * _dist(p1,p4) + 1e-6)

def _extract_points(landmarks, shape, idxs):
    h, w = shape[:2]
    return [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]

def _emotion_from_mesh(landmarks, shape):
    le = _extract_points(landmarks, shape, LEFT_EYE)
    re = _extract_points(landmarks, shape, RIGHT_EYE)
    ear = ( _ear(le) + _ear(re) ) / 2.0

    mouth_lr = _extract_points(landmarks, shape, MOUTH_W)
    mouth_tb = _extract_points(landmarks, shape, MOUTH_V)
    eye_outer = _extract_points(landmarks, shape, EYE_OUTER)

    mouth_width = _dist(mouth_lr[0], mouth_lr[1])
    mouth_open  = _dist(mouth_tb[0], mouth_tb[1])
    inter_ocular = _dist(eye_outer[0], eye_outer[1])

    smile_score = mouth_width / (inter_ocular + 1e-6)
    open_score  = mouth_open  / (inter_ocular + 1e-6)

    if ear < 0.17:
        return "Tired", {"EAR": round(ear,3), "Smile": round(smile_score,3), "Open": round(open_score,3)}
    if smile_score > 0.42 and open_score > 0.06:
        return "Happy", {"EAR": round(ear,3), "Smile": round(smile_score,3), "Open": round(open_score,3)}
    return "Neutral", {"EAR": round(ear,3), "Smile": round(smile_score,3), "Open": round(open_score,3)}

def face_scan(index=0):
    set_status("Scanning face…")
    frame = get_camera_frame(index)
    set_status("")
    if frame is None:
        return None, None, "Unknown", "No camera signal."

    if CAMERA_MIRROR:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)

    extras = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 55: extras.append("Low lighting")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 55: extras.append("Possible blur")

    if not res.detections:
        extras.append("No face detected")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(tempfile.gettempdir(), f"emma_facescan_{ts}.jpg")
        cv2.imwrite(out, frame)
        note = ", ".join(extras) if extras else "Looks OK"
        return out, _cv2_to_surface(frame), "Unknown", note

    # largest face
    best = None; best_area = -1
    for det in res.detections:
        bb = det.location_data.relative_bounding_box
        x, y, ww, hh = int(bb.xmin * w), int(bb.ymin * h), int(bb.width * w), int(bb.height * h)
        area = max(0, ww) * max(0, hh)
        if area > best_area:
            best_area = area
            best = (x, y, ww, hh)
    x, y, ww, hh = best
    x = max(0, x); y = max(0, y)
    ww = min(ww, w - x); hh = min(hh, h - y)

    # Landmarks for emotion
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        resm = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    emotion = "Neutral"
    if resm.multi_face_landmarks:
        fl = resm.multi_face_landmarks[0].landmark
        emotion, scores = _emotion_from_mesh(fl, frame.shape)
        extras.append(f"EAR:{scores['EAR']}, Smile:{scores['Smile']}, Open:{scores['Open']}")

    cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 180), 2)
    if resm.multi_face_landmarks:
        pts = [33, 133, 362, 263, 61, 291, 13, 14]
        for i in pts:
            p = resm.multi_face_landmarks[0].landmark[i]
            cv2.circle(frame, (int(p.x*w), int(p.y*h)), 2, (255, 220, 80), -1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(tempfile.gettempdir(), f"emma_facescan_{ts}.jpg")
    cv2.imwrite(out, frame)
    note = ", ".join(extras) if extras else "Looks OK"
    return out, _cv2_to_surface(frame), emotion, note
# ======= End Face Scan =======


# ======= Image Search (NEW) =======
def _download_image_to_surface(url):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        return img, _cv2_to_surface(img)
    except Exception:
        return None, None

def _wiki_search_first_title(query):
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":query,"format":"json","srlimit":1},
            timeout=15
        )
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        if hits:
            return hits[0]["title"]
    except Exception:
        pass
    return None

def _wiki_original_image(title):
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","titles":title,"prop":"pageimages","format":"json","piprop":"original"},
            timeout=15
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for _, p in pages.items():
            orig = p.get("original", {})
            if "source" in orig:
                return orig["source"]
    except Exception:
        pass
    return None

def _duckduckgo_vqd(query):
    try:
        html = requests.get("https://duckduckgo.com/", params={"q": query}, timeout=15).text
        m = re.search(r'vqd=([\d-]+)\&', html)
        if not m:
            m = re.search(r'vqd=([a-f0-9-]{10,})', html)
        return m.group(1) if m else None
    except Exception:
        return None

def _duckduckgo_first_image(query):
    vqd = _duckduckgo_vqd(query)
    if not vqd: return None
    try:
        js = requests.get(
            "https://duckduckgo.com/i.js",
            params={"l":"us-en","o":"json","q":query,"vqd":vqd,"f":"","p":"1"},
            timeout=15,
            headers={"referer":"https://duckduckgo.com/"}
        ).json()
        results = js.get("results", [])
        if results:
            return results[0].get("image") or results[0].get("thumbnail")
    except Exception:
        pass
    return None

def search_and_fetch_image(query):
    """Try Wikipedia first (good for people/places), then DuckDuckGo; return (surface, caption)."""
    # 1) Wikipedia
    title = _wiki_search_first_title(query)
    if title:
        url = _wiki_original_image(title)
        if url:
            img, surf = _download_image_to_surface(url)
            if surf is not None:
                return surf, f"{title} — from Wikipedia"

    # 2) DuckDuckGo Images
    url = _duckduckgo_first_image(query)
    if url:
        img, surf = _download_image_to_surface(url)
        if surf is not None:
            return surf, f"Top image result for “{query}”"

    # 3) Fallback: open images in browser
    webbrowser.open(f"https://duckduckgo.com/?iax=images&ia=images&q={query.replace(' ','+')}")
    return None, "I opened an image search in your browser."
# ======= End Image Search =======


def normalize_url(s):
    s = s.strip()
    if not s.lower().startswith(("http://", "https://")):
        s = "https://" + s
    return s

def open_website(text_or_url):
    q = text_or_url.strip()
    if re.match(r"^https?://", q) or "." in q:
        webbrowser.open(normalize_url(q)); return f"Opening {q}."
    webbrowser.open(f"https://www.google.com/search?q={q.replace(' ', '+')}"); return f"Searching the web for “{q}”."

def open_youtube_search(query=None):
    if not query:
        webbrowser.open("https://www.youtube.com/"); return "Opening YouTube."
    webbrowser.open(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"); return f"Searching YouTube for “{query}”."

def open_youtube_top_video(query):
    if not HAVE_YTDLP:
        return open_youtube_search(query)
    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if "entries" in info and info["entries"]:
                url = f"https://www.youtube.com/watch?v={info['entries'][0]['id']}"
                webbrowser.open(url); return f"Playing “{query}” on YouTube."
    except Exception:
        pass
    return open_youtube_search(query)

def play_online_mp3(url):
    set_status("Fetching MP3…")
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        path = os.path.join(tempfile.gettempdir(), f"emma_audio_{int(time.time())}.mp3")
        with open(path, "wb") as f: f.write(r.content)
        if pygame.mixer.get_init():
            try: pygame.mixer.music.stop()
            except Exception: pass
            pygame.mixer.music.load(path); pygame.mixer.music.play()
        return "Playing your MP3."
    except Exception as e:
        return f"Sorry, couldn’t play that MP3. ({e})"

def stop_music():
    try:
        if pygame.mixer.get_init(): pygame.mixer.music.stop()
        return "Stopped the music."
    except Exception:
        return "No music playing."


# ---------------- Intent router ----------------
def handle_intents(user_text):
    """
    Returns (handled: bool, reply_text: str). May push image bubbles directly.
    """
    raw = user_text.strip()
    t = raw.lower()

    # --- IMAGE/PHOTO SEARCH (NEW) ---
    m = re.search(r"(?:please\s+)?(?:find|search|show)\s+(.+?)\s+(?:image|images|photo|photos|picture|pictures)$", t)
    if m:
        q = re.sub(r"[-_]+", " ", m.group(1)).strip()
        add_message("system", f"Searching image: {q}")
        surf, cap = search_and_fetch_image(q)
        if surf is not None:
            add_image_message("emma", surf, caption=cap)
            return True, f"Here’s an image for “{q}”."
        else:
            return True, cap  # usually: opened a browser

    # --- PHOTO ---
    if re.search(r"(capturing image|capture image|take my picture|capture my photo|shot me photo)", t):
        add_message("system", "Capturing Photo")
        path, surf = capture_with_countdown(seconds=0)
        if surf is not None:
            add_image_message("emma", surf, caption="Captured photo")
            return True, "Here’s your photo."
        return True, "I couldn’t access the camera."

    # --- FACE SCAN ---
    if re.search(r"(scan(ning)? face|scan file face)", t):
        add_message("system", "Scanning Face")
        path, surf, emotion, note = face_scan()
        if surf is not None:
            add_image_message("emma", surf, caption=f"Emotion: {emotion} • {note}")
            return True, f"I scanned your face: {emotion}. {note}."
        return True, "I couldn’t access the camera for face scan."

    # --- YOUTUBE SEARCH ---
    m = re.search(r"^(search|open)\s+(.+?)\s+(?:in|on)\s+youtube$", t)
    if m:
        query = re.sub(r"[-_]+", " ", m.group(2).strip())
        return True, open_youtube_search(query)

    # --- PLAY MUSIC VIDEO / PLAY <name> ---
    m = re.search(r"^play\s+(?:music\s+video\s+)?(.+?)\s+(?:on|from)\s+youtube$", t)
    if m:
        query = re.sub(r"[-_]+", " ", m.group(1).strip())
        return True, open_youtube_top_video(query)

    m = re.search(r"^play\s+music\s+video\s+(.+)$", t)
    if m:
        query = re.sub(r"[-_]+", " ", m.group(1).strip())
        return True, open_youtube_top_video(query)

    if t.startswith("play "):
        if not re.search(r"(mp3|song|music\s+video)", t):
            query = re.sub(r"^play\s+", "", raw, flags=re.IGNORECASE).strip()
            query = re.sub(r"[-_]+", " ", query)
            return True, open_youtube_top_video(query)

    if t in ("open youtube", "go to youtube", "youtube"):
        return True, open_youtube_search(None)

    # --- OPEN/PLAY ONLINE MP3 ---
    if re.search(r"^(open|play)\s+(the\s+)?music\s+", t) or t.startswith("play mp3 "):
        murl = re.search(r"(https?://\S+\.mp3)", raw, flags=re.IGNORECASE)
        if murl:
            return True, play_online_mp3(murl.group(1))
        q = re.sub(r"^(open|play)\s+(the\s+)?music\s+", "", raw, flags=re.IGNORECASE).strip()
        if not q:
            q = re.sub(r"^play mp3\s+", "", raw, flags=re.IGNORECASE).strip()
        if q:
            webbrowser.open(f"https://www.google.com/search?q={q.replace(' ', '+')}+mp3")
            open_youtube_search(q)
            return True, f"Searching for “{q}” (mp3 + YouTube)."
        return True, "Tell me the song or give a direct .mp3 link."

    d = re.search(r"(https?://\S+\.mp3)", raw, flags=re.IGNORECASE)
    if d:
        return True, play_online_mp3(d.group(1))

    if t.startswith("open ") or t.startswith("go to "):
        rest = re.sub(r"^(open|go to)\s+", "", raw.strip(), flags=re.IGNORECASE).strip()
        if rest:
            return True, open_website(rest)

    if "stop music" in t or "pause music" in t:
        return True, stop_music()
    if "what time is it" in t or "current time" in t:
        return True, f"It’s {datetime.now().strftime('%H:%M:%S')}."
    return False, ""


# ---------------- Voice / AI / Servo ----------------
def listen_with_vosk(recognizer):
    set_status("Listening… (speak now)"); play_sfx("listen")
    mic = pyaudio.PyAudio()
    try:
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000,
                          input=True, frames_per_buffer=8192)
        stream.start_stream()
    except Exception as e:
        add_message("system", f"Microphone error: {e}")
        return ""
    try:
        while not shutdown_flag and mic_enabled:
            try:
                data = stream.read(8192, exception_on_overflow=False)
            except IOError:
                continue
            if len(data) == 0:
                continue
            if recognizer.AcceptWaveform(data):
                play_sfx("convert")
                result = recognizer.Result()
                return (json.loads(result).get("text") or "").strip()
        return ""
    finally:
        try:
            stream.stop_stream(); stream.close()
        except Exception:
            pass
        try:
            mic.terminate()
        except Exception:
            pass


def ollama_reply(user_text: str) -> str:
    set_status("Thinking…")
    try:
        # Add user with follow-up awareness
        CM.on_user(user_text)
        payload = {
            "model": OLLAMA_MODEL,
            "messages": CM.payload(),
            "stream": False
        }
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        msg = ((data.get("message") or {}).get("content") or "").strip()
        if not msg:
            msg = "I couldn’t generate a reply. Could you rephrase?"
        # Record assistant for continuity
        CM.on_assistant(msg)
        return msg
    except Exception as e:
        return f"Answering engine error: {e}. Is Ollama running and `{OLLAMA_MODEL}` pulled?"

def tts_stream(text):
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
        buf.seek(0); return buf
    except Exception as e:
        add_message("system", f"TTS error: {e}")
        return None

def play_audio(stream):
    global speaking_flag
    if stream is None:
        return None
    try:
        if not pygame.mixer.get_init():
            speaking_flag = True; return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(stream.read()); path = f.name
        speaking_flag = True
        pygame.mixer.music.load(path); pygame.mixer.music.play()
        return path
    except Exception as e:
        speaking_flag = True
        add_message("system", f"Audio playback error: {e}")
        return None

def move_servo(target_positions, delay=0.0009):
    global last_positions
    with position_lock:
        diffs = [abs(target_positions[i] - last_positions[i]) for i in range(3)]
        steps = max(diffs) if diffs else 0
        if steps == 0: return
        for st in range(steps):
            cur = []
            for i in range(3):
                start, end = last_positions[i], target_positions[i]
                cur.append(int(start + (st + 1) * (end - start) // steps))
            try: arduino.sendData(cur)
            except Exception: pass
            sleep(delay)
        last_positions = target_positions[:]

def presentation_movement():
    global presentation_active, last_positions
    dir_head = 1; dir_hand = 1
    min_head, max_head = 60, 120
    min_hand, max_hand = 0, 90
    step_head, step_hand = 6, 10
    while presentation_active and not shutdown_flag:
        with position_lock:
            head = last_positions[2] + dir_head * step_head
            hand = last_positions[1] + dir_hand * step_hand
        if head > max_head or head < min_head:
            dir_head *= -1; head = max(min_head, min(max_head, head))
        if hand > max_hand or hand < min_hand:
            dir_hand *= -1; hand = max(min_hand, min(max_hand, hand))
        move_servo([last_positions[0], int(hand), int(head)], delay=0.005)
        sleep(0.18)

def hello_gesture():
    move_servo([last_positions[0], 180, last_positions[2]])
    for _ in range(3):
        move_servo([last_positions[0], 150, last_positions[2]])
        move_servo([last_positions[0], 180, last_positions[2]])
    move_servo([last_positions[0], 0, last_positions[2]])

def voice_worker():
    try:
        model = vosk.Model(VOSK_MODEL_PATH)
        recog = vosk.KaldiRecognizer(model, 16000)
    except Exception as e:
        add_message("system", f"Vosk model load error: {e}")
        return

    try:
        move_servo(HOME_POS, delay=0.001)
    except Exception:
        pass

    while not shutdown_flag:
        if not user_text_queue.empty():
            user_text = user_text_queue.get()
        elif mic_enabled:
            user_text = listen_with_vosk(recog)
            if user_text:
                add_message("you", user_text)
        else:
            time.sleep(0.05)
            continue

        if not user_text:
            continue

        handled, msg = handle_intents(user_text)
        if handled:
            add_message("emma", msg)
            last_txt = msg
        else:
            if "hello" in user_text.lower() or "emma" in user_text.lower():
                try: hello_gesture()
                except Exception: pass
                last_txt = "Hello! How can I assist you today?"
            else:
                last_txt = ollama_reply(user_text)   # <-- answers via Ollama
            add_message("emma", last_txt)

        # talking + motion
        global presentation_active
        presentation_active = True
        m = threading.Thread(target=presentation_movement, daemon=True); m.start()

        if last_txt:
            set_status("Speaking…")
            play_audio(tts_stream(last_txt))
            if pygame.mixer.get_init():
                while pygame.mixer.music.get_busy() and not shutdown_flag:
                    time.sleep(0.03)
            else:
                time.sleep(1.0)
            set_status("")

        presentation_active = False
        try:
            if m.is_alive(): m.join(timeout=1.0)
        except Exception:
            pass

        try:
            move_servo(HOME_POS, delay=0.001)
        except Exception:
            pass


# ---------------- Main loop ----------------
def ui_main_loop():
    global shutdown_flag, input_text, mic_enabled
    global speaking_flag, chat_scroll_px, auto_scroll_to_bottom
    global scroll_dragging, scroll_drag_dy

    ui_init()
    if not chat_messages:
        add_message("system", "Ready. Use F2 to toggle mic, or type and press Enter.")
        add_message("emma", "Hello, I am EMMA. How can I assist you?")

    threading.Thread(target=voice_worker, daemon=True).start()

    while not shutdown_flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shutdown_flag = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    shutdown_flag = True
                elif event.key == pygame.K_F2:
                    mic_enabled = not mic_enabled
                    set_status(f"Mic {'enabled' if mic_enabled else 'disabled'}.")
                elif event.key == pygame.K_PAGEUP and _last_chat_rect:
                    set_chat_scroll(+140, _last_chat_rect)
                elif event.key == pygame.K_PAGEDOWN and _last_chat_rect:
                    set_chat_scroll(-140, _last_chat_rect)
                elif event.key == pygame.K_HOME and _last_chat_rect:
                    auto_scroll_to_bottom = False
                    max_scroll = max(0, chat_content_height - _last_chat_rect.height + 20)
                    set_chat_scroll(max_scroll - chat_scroll_px, _last_chat_rect)
                elif event.key == pygame.K_END and _last_chat_rect:
                    set_chat_scroll("bottom", _last_chat_rect)
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]; auto_scroll_to_bottom = True
                elif event.key == pygame.K_RETURN:
                    txt = input_text.strip()
                    if txt:
                        add_message("you", txt)
                        user_text_queue.put(txt)
                        input_text = ""
                        auto_scroll_to_bottom = True
                else:
                    if event.unicode and event.unicode.isprintable():
                        input_text += event.unicode
                        auto_scroll_to_bottom = True

            elif event.type == pygame.MOUSEWHEEL and _last_chat_rect:
                if _last_chat_rect.collidepoint(pygame.mouse.get_pos()):
                    set_chat_scroll(+ (80 if event.y > 0 else -80), _last_chat_rect)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if _last_scroll_thumb and _last_scroll_thumb.collidepoint(event.pos):
                    scroll_dragging = True
                    scroll_drag_dy = event.pos[1] - _last_scroll_thumb.y
                elif _last_scroll_track and _last_scroll_track.collidepoint(event.pos):
                    tr, th = _last_scroll_track, _last_scroll_thumb
                    hh = th.height
                    y = min(max(event.pos[1] - tr.y - hh // 2, 0), tr.height - hh)
                    view_h = _last_chat_rect.height - 4
                    max_scroll = max(0, chat_content_height - view_h)
                    frac = 0.0 if tr.height - hh == 0 else (1.0 - y / float(tr.height - hh))
                    new_px = int(max_scroll * (1.0 - frac))
                    set_chat_scroll(new_px - chat_scroll_px, _last_chat_rect)
                    auto_scroll_to_bottom = (chat_scroll_px == 0)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                scroll_dragging = False

            elif event.type == pygame.MOUSEMOTION and scroll_dragging and _last_scroll_track:
                tr = _last_scroll_track; hh = _last_scroll_thumb.height
                y = min(max(event.pos[1] - tr.y - scroll_drag_dy, 0), tr.height - hh)
                view_h = _last_chat_rect.height - 4
                max_scroll = max(0, chat_content_height - view_h)
                frac = 0.0 if tr.height - hh == 0 else (1.0 - y / float(tr.height - hh))
                new_px = int(max_scroll * (1.0 - frac))
                set_chat_scroll(new_px - chat_scroll_px, _last_chat_rect)
                auto_scroll_to_bottom = (chat_scroll_px == 0)

        # draw
        vgrad(screen, (0, 0, UI_WIDTH, UI_HEIGHT), (8, 14, 28), (10, 44, 84))
        draw_header()

        left = pygame.Rect(PAD, PAD + HEADER_H + PAD, SIDEBAR_W, UI_HEIGHT - (HEADER_H + 3 * PAD))
        right = pygame.Rect(PAD * 2 + SIDEBAR_W, PAD + HEADER_H + PAD, UI_WIDTH - (SIDEBAR_W + 3 * PAD), UI_HEIGHT - (HEADER_H + 3 * PAD))

        dock = pygame.Surface((left.w, left.h), pygame.SRCALPHA)
        vgrad(dock, (0, 0, left.w, left.h), (14, 26, 44), (10, 22, 38))
        pygame.draw.rect(dock, (160, 220, 255, 70), (0, 0, left.w, left.h), 1, border_radius=16)
        screen.blit(dock, (left.x, left.y))

        draw_eve(left.centerx, left.y + int(left.h * 0.54), height_px=int(left.h * 0.92), t=time.time())
        draw_chat_and_input(right.x, right.y, right.w, right.h)

        pygame.display.flip()
        clock.tick(UI_FPS)

        if speaking_flag and (not pygame.mixer.get_init() or not pygame.mixer.music.get_busy()):
            speaking_flag = False

    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception:
        pass
    pygame.quit()


# ---------------- Entry ----------------
if __name__ == "__main__":
    try:
        arduino = SerialObject(digits=ARDUINO_DIGITS)
    except Exception:
        class DummySerial:
            def sendData(self, *_a, **_k): pass
        arduino = DummySerial()

    try:
        ui_main_loop()
    except KeyboardInterrupt:
        shutdown_flag = True
