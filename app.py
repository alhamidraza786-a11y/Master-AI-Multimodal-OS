import json
import os
import io
import asyncio
import base64
import concurrent.futures
import urllib.parse
import requests
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
from langchain_groq import ChatGroq
from groq import Groq
from gtts import gTTS
from PIL import Image, ImageDraw
import edge_tts

# ==========================================
# 0. Config Loader
# ==========================================
def get_config(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

# ==========================================
# 1. Google Drive & PDF
# ==========================================
def get_google_drive_service():
    sa = get_config("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa:
        return None
    try:
        cd = json.loads(sa) if isinstance(sa, str) else sa
        scopes = ['https://www.googleapis.com/auth/drive.readonly']
        creds = service_account.Credentials.from_service_account_info(cd, scopes=scopes)
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"Drive error: {e}")
        return None

KEYWORDS_DRIVE = ["drive", "google drive", "book", "library", "pdf", "kitab", "kitaab",
                  "ڈرائیو", "کتاب", "کتابیں", "لائبریری", "پی ڈی ایف", "فائل"]
KEYWORDS_IMAGE = ["image", "picture", "graphic", "infographic", "poster", "banner", "diagram",
                  "تصویر", "ڈیزائن", "انفوگرافک", "پوسٹر", "بینر", "گرافک", "نقشہ", "خاکہ"]
KEYWORDS_VOICE = ["voice", "audio", "sound", "speak", "speech", "awaaz", "bol", "parh",
                  "آواز", "وائس", "آڈیو", "بولیں", "سنائیں", "پڑھیں"]

def has_kw(text, kw_list):
    if not text:
        return False
    t = text.lower()
    return any(k.lower() in t for k in kw_list)

STOPWORDS = {"کا","کی","کے","ہے","ہیں","میں","سے","کو","اور","پر","ایک","کیا","کیسے",
             "کیوں","کہاں","کون","کب","کہ","نے","نہ","بھی","تو","ہی","یہ","وہ","اس","ان",
             "the","a","an","is","are","was","were","of","in","on","at","to","from","for","with",
             "by","about","as","into","and","or","but","if","then","what","which","who","this"}

def extract_words(query):
    if not query:
        return []
    for ch in "؟?.,!،:;\"'()[]{}":
        query = query.replace(ch, " ")
    return [w for w in query.split() if len(w) > 2 and w.lower() not in STOPWORDS]

@st.cache_data(ttl=300, show_spinner=False)
def search_drive_books(user_query, max_files=10, max_pages=8):
    service = get_google_drive_service()
    if not service:
        return None, "drive_unavailable"
    try:
        words = extract_words(user_query)
        results = service.files().list(
            q="mimeType='application/pdf'", pageSize=max_files,
            fields="files(id, name, mimeType)"
        ).execute()
        files = results.get('files', [])
        if not files:
            return None, "no_files"
        out = "📚 Google Drive Books:\n\n"
        found = False
        for f in files:
            fname = f['name']
            name_match = any(w.lower() in fname.lower() for w in words) if words else True
            try:
                req = service.files().get_media(fileId=f['id'])
                fh = io.BytesIO()
                dl = MediaIoBaseDownload(fh, req)
                done = False
                while not done:
                    _, done = dl.next_chunk()
                fh.seek(0)
                reader = PdfReader(fh)
                pdf_text = ""
                for i, page in enumerate(reader.pages):
                    if i >= max_pages:
                        break
                    try:
                        pdf_text += (page.extract_text() or "") + "\n"
                    except Exception:
                        pass
                content_match = any(w.lower() in pdf_text.lower() for w in words) if words else False
                if name_match or content_match:
                    found = True
                    out += f"\n### 📄 {fname}\n"
                    if content_match and words:
                        for w in words:
                            idx = pdf_text.lower().find(w.lower())
                            if idx > 0:
                                out += f"*'{w}' سے متعلق:*\n...{pdf_text[max(0,idx-300):idx+800]}...\n\n"
                                break
                    else:
                        out += f"...{pdf_text[:1200]}...\n\n"
            except Exception:
                continue
        return (out, "found") if found else (None, "not_relevant")
    except Exception as e:
        return None, f"error:{e}"

# ==========================================
# 2. Image Generation (WITH TRANSLATION FIX)
# ==========================================
def generate_ai_image(prompt_text, style="infographic", width=1024, height=600):
    style_map = {
        "infographic": "educational HD infographic poster, clear labels, medical graphics",
        "realistic": "photorealistic, 8k, highly detailed, professional photography",
        "3d_avatar": "3D avatar character, pixar style, ultra detailed",
        "cartoon": "cartoon illustration, vibrant colors, cute style",
        "game_asset": "game asset, isometric, 3D render",
        "web_design": "modern website UI design, clean layout",
        "logo": "professional logo design, minimal, vector",
        "anime": "anime style, studio ghibli",
        "cyberpunk": "cyberpunk aesthetic, neon lights",
        "sketch": "pencil sketch, hand drawn",
        "engineering": "technical engineering diagram, blueprint style",
    }
    full_prompt = urllib.parse.quote(f"{prompt_text}, {style_map.get(style, '')}")
    url = f"https://image.pollinations.ai/prompt/{full_prompt}?width={width}&height={height}&nologo=true&model=flux"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 1024:
            return io.BytesIO(r.content)
    except Exception as e:
        print(f"Image error: {e}")
    return None

def fallback_image(title, size=(900, 500)):
    try:
        img = Image.new('RGB', size, color='#0f172a')
        d = ImageDraw.Draw(img)
        d.rectangle([(20, 20), (size[0]-20, size[1]-20)], outline='#38bdf8', width=4)
        d.text((50, 50), f"AI: {title[:50]}", fill='#f8fafc')
        d.text((50, 160), "Master AI Engine", fill='#cbd5e1')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
    except Exception:
        return None

# ==========================================
# 3. Voice Engine
# ==========================================
EDGE_VOICES = {
    "اردو / Urdu": {"اسد (M)": "ur-PK-AsadNeural", "عظمیٰ (F)": "ur-PK-UzmaNeural"},
    "English-US": {"Aria (F)": "en-US-AriaNeural", "Jenny (F)": "en-US-JennyNeural",
                   "Guy (M)": "en-US-GuyNeural", "Tony (M)": "en-US-TonyNeural"},
    "English-UK": {"Sonia (F)": "en-GB-SoniaNeural", "Ryan (M)": "en-GB-RyanNeural"},
    "ہندی / Hindi": {"Swara (F)": "hi-IN-SwaraNeural", "Madhur (M)": "hi-IN-MadhurNeural"},
    "عربی / Arabic": {"Zariyah (F)": "ar-SA-ZariyahNeural", "Hamed (M)": "ar-SA-HamedNeural"},
    "پنجابی / Punjabi": {"Uzma (F)": "pa-PK-UzmaNeural", "Asad (M)": "pa-PK-AsadNeural"},
    "فارسی / Persian": {"Dilara (F)": "fa-IR-DilaraNeural", "Farid (M)": "fa-IR-FaridNeural"},
    "چینی / Chinese": {"Xiaoxiao (F)": "zh-CN-XiaoxiaoNeural", "Yunxi (M)": "zh-CN-YunxiNeural"},
    "فرنچ / French": {"Denise (F)": "fr-FR-DeniseNeural", "Henri (M)": "fr-FR-HenriNeural"},
    "جرمن / German": {"Katja (F)": "de-DE-KatjaNeural", "Conrad (M)": "de-DE-ConradNeural"},
    "ہسپانوی / Spanish": {"Elvira (F)": "es-ES-ElviraNeural", "Alvaro (M)": "es-ES-AlvaroNeural"},
    "روسی / Russian": {"Svetlana (F)": "ru-RU-SvetlanaNeural", "Dmitry (M)": "ru-RU-DmitryNeural"},
}

VOICE_SPEEDS = {"🐢 بہت آہستہ": "-40%", "🚶 آہستہ": "-20%", "⚡ نارمل": "+0%",
                "🏃 تیز": "+30%", "🚀 بہت تیز": "+50%"}
VOICE_PITCHES = {"🔽 گہری": "-15Hz", "▶️ نارمل": "+0Hz", "🔼 اونچی": "+15Hz"}

def detect_lang(text):
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return 'ur'
    if any('\u0900' <= c <= '\u097F' for c in text):
        return 'hi'
    return 'en'

async def _edge_async(text, voice, rate, pitch):
    c = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    buf = bytearray()
    async for chunk in c.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)

def generate_voice_edge(text, voice, rate="+0%", pitch="+0Hz"):
    def _thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_edge_async(text, voice, rate, pitch))
        finally:
            loop.close()
    try:
        clean = text.replace("*", "").replace("#", "").replace("`", "").strip()
        if not clean:
            return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            audio = ex.submit(_thread).result(timeout=60)
        if audio:
            return io.BytesIO(audio)
    except Exception as e:
        print(f"Edge error: {e}")
    return None

def generate_voice_gtts(text, lang=None):
    try:
        clean = text.replace("*", "").replace("#", "").replace("`", "")
        if not clean.strip():
            return None
        if not lang:
            lang = detect_lang(clean)
        tts = gTTS(text=clean[:500], lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None

def create_voice(text, voice_id=None, rate="+0%", pitch="+0Hz"):
    if voice_id:
        a = generate_voice_edge(text, voice_id, rate, pitch)
        if a:
            return a
    return generate_voice_gtts(text)

# ==========================================
# 4. YouTube, Whisper, OCR, PDF
# ==========================================
def extract_yt_id(url):
    try:
        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        if "shorts/" in url:
            return url.split("shorts/")[1].split("?")[0]
        return url.strip()
    except Exception:
        return None

def get_yt_transcript(url, langs=("ur", "hi", "en")):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        vid = extract_yt_id(url)
        if not vid:
            return None, "ID نہیں ملی"
        try:
            tl = YouTubeTranscriptApi.list_transcripts(vid)
        except Exception as e:
            return None, f"دستیاب نہیں: {e}"
        chosen = None
        for l in langs:
            try:
                chosen = tl.find_transcript([l])
                break
            except Exception:
                continue
        if not chosen:
            try:
                chosen = tl.find_generated_transcript(list(langs))
            except Exception:
                for t in tl:
                    chosen = t
                    break
        if not chosen:
            return None, "کوئی ٹرانسکرپٹ نہیں"
        data = chosen.fetch()
        return " ".join([i['text'] for i in data]), vid
    except Exception as e:
        return None, f"Error: {e}"

def transcribe_whisper(audio_bytes, filename, language=None):
    try:
        client = Groq(api_key=groq_api_key)
        kw = {"file": (filename, audio_bytes), "model": "whisper-large-v3-turbo", "response_format": "text"}
        if language and language != "auto":
            kw["language"] = language
        t = client.audio.transcriptions.create(**kw)
        return t if isinstance(t, str) else str(t)
    except Exception as e:
        return f"⚠️ {e}"

def ocr_image(image_bytes, extra=""):
    try:
        client = Groq(api_key=groq_api_key)
        b64 = base64.b64encode(image_bytes).decode()
        mime = "image/jpeg"
        if image_bytes[:8].startswith(b'\x89PNG'):
            mime = "image/png"
        elif image_bytes[:4] == b'RIFF':
            mime = "image/webp"
        prompt = ("Extract ALL text from this image exactly as it appears. "
                  "Preserve original language (Urdu→Urdu, English→English). No commentary. " + extra)
        chat = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]}],
            temperature=0.1, max_tokens=2048,
        )
        return chat.choices[0].message.content
    except Exception as e:
        return f"⚠️ {e}"

def extract_pdf_text(pdf_bytes, max_pages=20):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        t = ""
        for i, p in enumerate(reader.pages):
            if i >= max_pages:
                break
            try:
                t += (p.extract_text() or "") + "\n"
            except Exception:
                pass
        return t.strip()
    except Exception as e:
        return f"PDF error: {e}"

# ==========================================
# 5. Streamlit Config
# ==========================================
st.set_page_config(page_title="Master AI Multimodal OS", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 6. UNIVERSAL SECTIONS CONFIG (76 Sections)
# ==========================================
SECTIONS_CONFIG = {
    # Medical
    "🏥 ہیومن میڈیسن": ("امراض، علامات، علاج، نسخے",
        [("ta", "علامات / Symptoms", "sym", "مثلاً: بخار، کھانسی"),
         ("tx", "عمر / Age", "age", "مثلاً: 35"),
         ("se", "جنس / Gender", "gen", ["مرد/Male", "خاتون/Female", "بچہ/Child"])],
        "senior medical doctor providing diagnosis, treatment, prescriptions with dosage"),
    "🐄 ویٹرنری میڈیسن": ("جانوروں کی بیماریاں، علاج",
        [("tx", "جانور / Animal", "animal", "گائے، بکری، کتا"),
         ("ta", "علامات / Symptoms", "sym", ""),
         ("tx", "عمر / Age", "age", "")],
        "expert veterinarian providing animal diagnosis, treatment, surgery guidance"),
    "💊 فارماسی / دوائی": ("دوائیں، ڈوز، تعامل",
        [("tx", "دوائی / Medicine", "med", ""),
         ("ta", "استعمال / Purpose", "purp", "")],
        "clinical pharmacist explaining medicine, dosage, interactions, side effects"),
    "🧬 جینیٹکس / DNA": ("وارثت، جین",
        [("ta", "سوال / Question", "q", "")],
        "geneticist explaining inheritance, DNA, mutations, genetic disorders"),
    "🔬 بائیو ٹیکنالوجی": ("جینیٹکس، لیب ورک",
        [("ta", "موضوع / Topic", "t", "")],
        "biotechnologist covering lab techniques, genetic engineering, CRISPR"),
    # Engineering
    "🏗️ سول انجینئرنگ": ("عمارتیں، پل، سڑکیں",
        [("ta", "منصوبہ / Project", "p", ""),
         ("tx", "مقام / Location", "loc", "")],
        "civil engineer providing structural design, concrete, foundations, ACI/ASCE standards"),
    "🧪 کیمیکل انجینئرنگ": ("کیمیائی عمل، پلانٹس",
        [("ta", "موضوع / Topic", "t", "")],
        "chemical engineer explaining processes, reactors, distillation, mass/energy balance"),
    "✈️ ایرو اسپیس انجینئرنگ": ("ہوائی جہاز، راکٹ",
        [("ta", "موضوع / Topic", "t", "")],
        "aerospace engineer covering aerodynamics, propulsion, structures, orbital mechanics"),
    "⚡ الیکٹریکل انجینئرنگ": ("Power, Circuits, Machines",
        [("ta", "موضوع / Topic", "t", "")],
        "electrical engineer covering power systems, circuits, machines, electronics, standards"),
    "⚙️ مکینیکل انجینئرنگ": ("Thermo, Mechanics, Design",
        [("ta", "موضوع / Topic", "t", "")],
        "mechanical engineer covering thermodynamics, mechanics, machine design, manufacturing"),
    # Law & Finance
    "🏛️ وکیل / قانونی مشیر": ("قانونی معاملات، معاہدے",
        [("tx", "قسم / Type", "type", "معاہدہ، مقدمہ، ٹیکس"),
         ("ta", "تفصیل / Details", "d", "")],
        "legal advisor providing contract drafting, legal opinions, dispute resolution"),
    "🧮 اکاؤنٹنگ / ٹیکس": ("حساب، ٹیکس، بجٹ",
        [("tx", "قسم / Type", "type", "بک کیپنگ، ٹیکس، آڈٹ"),
         ("ta", "تفصیل / Details", "d", "")],
        "chartered accountant covering bookkeeping, financial statements, tax, audit, IFRS"),
    "📊 اسٹاک مارکیٹ / ٹریڈنگ": ("سرمایہ کاری، ٹریڈنگ",
        [("tx", "مارکیٹ / Market", "m", "PSX, NYSE, Crypto"),
         ("ta", "سوال / Question", "q", "")],
        "stock analyst providing market analysis, technical/fundamental analysis, strategies"),
    # Education
    "🎓 تعلیم / ٹیوٹر": ("کورسز، امتحانات، نوٹس",
        [("tx", "مضمون / Subject", "sub", ""),
         ("tx", "کلاس / Class", "cls", ""),
         ("ta", "سوال / Question", "q", "")],
        "expert tutor explaining concepts, solving problems, exam prep, study plans"),
    "🌍 ترجمہ / Translation": ("تمام زبانیں",
        [("se", "سے / From", "fr", ["Urdu", "English", "Arabic", "Hindi", "Persian", "Chinese"]),
         ("se", "تک / To", "to", ["English", "Urdu", "Arabic", "Hindi", "Persian", "Chinese"]),
         ("ta", "متن / Text", "txt", "")],
        "professional translator translating accurately while preserving meaning, tone"),
    "✍️ پروف ریڈنگ / ایڈیٹنگ": ("غلطیاں درست",
        [("ta", "متن / Text", "txt", "")],
        "professional editor proofreading and improving text: grammar, style, clarity"),
    "📚 کتاب / ناول لکھنا": ("کہانی، پلاٹ",
        [("tx", "قسم / Type", "type", "ناول، کہانی، شاعری"),
         ("ta", "خیال / Idea", "idea", "")],
        "expert author providing plot outline, character development, writing tips"),
    # Life
    "🌾 زراعت / کاشتکاری": ("فصلیں، کیڑے، کھاد",
        [("tx", "فصل / Crop", "crop", ""),
         ("ta", "سوال / Question", "q", "")],
        "agricultural expert covering crop management, pests, fertilizers, irrigation"),
    "🍳 کھانا پکانا / ریسپی": ("ریسپیز، ڈائٹ پلان",
        [("tx", "ڈش / Dish", "dish", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "master chef providing recipes, cooking techniques, ingredient substitutes, plating"),
    "💪 جم / فٹنس / ڈائٹ": ("ورزش، ڈائٹ چارٹ",
        [("tx", "مقصد / Goal", "goal", "Weight Loss, Muscle Gain, Fitness"),
         ("tx", "عمر / Age", "age", ""),
         ("tx", "قد/وزن / Height/Weight", "hw", "")],
        "certified fitness trainer and nutritionist providing workout plans, diet charts"),
    "🧘 یوگا / مراقبہ": ("یوگا، مراقبہ، روحانیت",
        [("tx", "مقصد / Goal", "goal", ""),
         ("ta", "سوال / Question", "q", "")],
        "yoga instructor and meditation guide teaching asanas, pranayama, mindfulness"),
    # Travel
    "✈️ ٹریول / سیاحت": ("ویزا، ٹکٹ، ہوٹل، ٹور",
        [("tx", "منزل / Destination", "dest", ""),
         ("tx", "بجٹ / Budget", "budget", ""),
         ("tx", "دن / Days", "days", "")],
        "travel expert providing itinerary, visa info, hotels, transport, budget planning"),
    # Creative
    "🎬 فلم / ڈرامہ اسکرپٹ": ("کہانی، اسکرپٹ",
        [("tx", "قسم / Type", "type", "فلم، ڈرامہ، اشتہار"),
         ("ta", "خیال / Idea", "idea", "")],
        "screenwriter providing complete script with scenes, dialogues, character arcs"),
    "🎵 موسیقی کمپوزیشن": ("دھن، راگ",
        [("tx", "اسٹائل / Style", "style", ""),
         ("tx", "موڈ / Mood", "mood", "")],
        "music composer providing melody, chords, rhythm, arrangement, instrumentation"),
    "🎮 گیم ڈویلپمنٹ": ("گیم آئیڈیا، ڈیزائن",
        [("tx", "قسم / Type", "type", "2D, 3D, Mobile, PC"),
         ("tx", "انجن / Engine", "eng", "Unity, Unreal, Godot"),
         ("ta", "خیال / Idea", "idea", "")],
        "game designer providing complete GDD, mechanics, levels, art style, sample code"),
    # Tech
    "🧠 AI / ML": ("AI ماڈلز، ٹریننگ",
        [("tx", "موضوع / Topic", "t", ""),
         ("se", "لیول / Level", "lvl", ["Beginner", "Intermediate", "Advanced"])],
        "AI/ML engineer covering models, training, deployment, Python code, TensorFlow"),
    "🤖 روبوٹکس / IoT": ("روبوٹ، سینسر",
        [("tx", "پروجیکٹ / Project", "p", ""),
         ("tx", "ہارڈویئر / Hardware", "hw", "Arduino, Raspberry Pi, ESP32")],
        "robotics/IoT engineer providing circuit, code, sensor integration, automation"),
    "🚀 اسٹارٹ اپ / بزنس پلان": ("بزنس آئیڈیا، پچ ڈیک",
        [("ta", "آئیڈیا / Idea", "idea", ""),
         ("tx", "مارکیٹ / Market", "market", "")],
        "startup advisor providing business model, pitch deck, financials, go-to-market"),
    "📊 ڈیٹا سائنس": ("ڈیٹا، گراف، پریڈکشن",
        [("tx", "موضوع / Topic", "t", ""),
         ("ta", "ڈیٹا / Data", "d", "")],
        "data scientist providing analysis, visualization, ML models, Python code"),
    "📊 ڈیٹا اینالیسز": ("ڈیٹا تجزیہ",
        [("ta", "ڈیٹا / Data", "d", ""),
         ("tx", "مقصد / Goal", "goal", "")],
        "data analyst providing statistical analysis, Excel/Python formulas, trends"),
    # Culture
    "🕌 اسلامیات / قرآن / حدیث": ("تفسیر، احکام",
        [("ta", "سوال / Question", "q", ""),
         ("tx", "موضوع / Topic", "t", "")],
        "Islamic scholar providing Quran tafsir, Hadith references, fiqh rulings"),
    "🏆 کھیل / کوچنگ": ("کرکٹ، فٹبال، کوچنگ",
        [("tx", "کھیل / Sport", "s", ""),
         ("ta", "سوال / Question", "q", "")],
        "sports coach providing training, techniques, strategy, fitness"),
    "🎯 سپورٹس پریڈکشن": ("میچ پریڈکشن",
        [("tx", "میچ / Match", "m", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "sports analyst providing match prediction, odds analysis, historical data"),
    "💼 جاب / انٹرویو": ("CV، انٹرویو تیاری",
        [("tx", "پوزیشن / Position", "pos", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "career coach providing CV, cover letter, interview prep, salary negotiation"),
    # Business
    "🏠 رئیل اسٹیٹ": ("پراپرٹی، انویسٹمنٹ",
        [("tx", "قسم / Type", "type", "خرید، فروخت، کرایہ"),
         ("tx", "مقام / Location", "loc", "")],
        "real estate expert providing property valuation, investment advice, legal"),
    "🛡️ بیمہ / انشورنس": ("پالیسی، کلیم",
        [("tx", "قسم / Type", "type", "Life, Health, Auto, Property"),
         ("ta", "تفصیل / Details", "d", "")],
        "insurance advisor explaining policies, coverage, claims, premium calculation"),
    "🧾 بلنگ / انوائس": ("بل بنانا",
        [("tx", "کلائنٹ / Client", "c", ""),
         ("ta", "اشیاء / Items", "it", "name, qty, price per line")],
        "accounting expert creating professional invoice with totals, tax, terms"),
    "📦 ای کامرس / ڈراپ شپنگ": ("Daraz، Amazon، Shopify",
        [("tx", "پلیٹ فارم / Platform", "p", "Daraz, Amazon, Shopify"),
         ("ta", "پروڈکٹ / Product", "prod", "")],
        "e-commerce expert providing product sourcing, listing optimization, ads"),
    "🚚 لاجسٹکس / شپنگ": ("ڈیلیوری",
        [("tx", "قسم / Type", "type", "Local, International"),
         ("ta", "تفصیل / Details", "d", "")],
        "logistics expert providing shipping, customs, freight, supply chain, warehouse"),
    "📊 ڈیٹا اینالٹکس": ("Business Analytics",
        [("ta", "ڈیٹا / Data", "d", ""),
         ("tx", "مقصد / Goal", "goal", "")],
        "business analyst providing KPIs, dashboards, insights, recommendations"),
    # Freelancing
    "💼 Upwork": ("Upwork Freelancing",
        [("se", "قسم / Type", "type", ["Hourly", "Fixed Price", "Both"]),
         ("tx", "نیش / Niche", "n", ""),
         ("ta", "سوال / Question", "q", "")],
        "Upwork expert providing profile optimization, proposals, hourly vs fixed strategy"),
    "💰 Upwork Hourly": ("Upwork Hourly Pay",
        [("tx", "نیش / Niche", "n", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "Upwork expert providing hourly rate strategy, time tracking, contracts, Top Rated tips"),
    "🎨 Fiverr": ("Fiverr Gigs",
        [("tx", "گیگ / Gig", "g", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "Fiverr expert providing gig optimization, pricing, buyer requests, level progression"),
    "⏰ PeoplePerHour": ("PPH Freelancing",
        [("tx", "نیش / Niche", "n", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "PeoplePerHour expert providing proposals, hourly, packages, client acquisition"),
    "💼 LinkedIn": ("LinkedIn Marketing",
        [("tx", "مقصد / Goal", "g", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "LinkedIn expert providing profile optimization, content strategy, lead generation, B2B"),
    "📸 Instagram": ("Instagram Marketing",
        [("tx", "نیش / Niche", "n", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "Instagram expert providing content plan, reels strategy, hashtags, growth"),
    "👻 Snapchat": ("Snapchat Marketing",
        [("tx", "نیش / Niche", "n", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "Snapchat expert providing marketing, AR lenses, ads, growth strategy"),
    "🛒 Etsy": ("Etsy Selling",
        [("tx", "پروڈکٹ / Product", "p", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "Etsy expert providing shop setup, SEO, product photos, pricing, promotion"),
    "📌 Pinterest Native": ("Pinterest Marketing",
        [("tx", "نیش / Niche", "n", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "Pinterest expert providing pins, boards, SEO, traffic strategy, monetization"),
    "🐦 Twitter Marketing": ("Twitter/X Marketing",
        [("tx", "نیش / Niche", "n", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "Twitter expert providing tweets, threads, engagement, growth, monetization"),
    # Affiliate & Marketing
    "🔗 Affiliate Marketing": ("ایفیلیٹ مارکیٹنگ",
        [("tx", "پروڈکٹ / Product", "p", ""),
         ("tx", "پلیٹ فارم / Platform", "plt", "Amazon, ClickBank, Digistore")],
        "affiliate marketer providing niche, offers, traffic, content, conversion optimization"),
    "🤝 B2B Marketing": ("Business to Business",
        [("tx", "انڈسٹری / Industry", "ind", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "B2B marketing expert providing ABM, LinkedIn outreach, sales funnels, lead gen"),
    "🛍️ B2C Marketing": ("Business to Consumer",
        [("tx", "پروڈکٹ / Product", "p", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "B2C marketing expert providing consumer funnels, social ads, retention, LTV"),
    "⚙️ Marketing Automation": ("مارکیٹنگ آٹو میشن",
        [("tx", "اوزار / Tool", "t", "HubSpot, Mailchimp, Zapier"),
         ("ta", "تفصیل / Details", "d", "")],
        "marketing automation expert providing workflows, email sequences, CRM, integrations"),
    "📝 Content Marketing": ("کونٹینٹ مارکیٹنگ",
        [("tx", "نیش / Niche", "n", ""),
         ("ta", "تفصیل / Details", "d", "")],
        "content marketer providing strategy, calendars, SEO content, distribution"),
    "🎬 3D Video Generation": ("تھری ڈی ویڈیو جنریٹنگ",
        [("tx", "قسم / Type", "type", "Explainer, Product, Character"),
         ("ta", "تفصیل / Details", "d", "")],
        "3D video expert providing script, storyboard, 3D workflow, tools, rendering"),
    # SEO & Ads
    "📈 SEO On-Page": ("ویب سائٹ آپٹیمائزیشن",
        [("tx", "URL", "url", ""),
         ("tx", "کی ورڈز / Keywords", "kw", "")],
        "SEO expert providing on-page optimization, meta tags, keywords, content strategy"),
    "🔗 SEO Off-Page + Backlinks": ("بیک لنکس، گیسٹ پوسٹنگ",
        [("tx", "ویب سائٹ / Website", "site", ""),
         ("tx", "نیش / Niche", "n", "")],
        "SEO expert providing backlink strategy, guest posting, 30+ link types, outreach"),
    "✍️ کاپی رائٹنگ": ("پروفیشنل کاپی",
        [("tx", "پروڈکٹ / Product", "p", ""),
         ("se", "لہجہ / Tone", "tone", ["پروفیشنل", "دوستانه", "جذباتی", "Persuasive"])],
        "professional copywriter providing headlines, ad copy, landing pages, email sequences"),
    "📱 سوشل میڈیا منیجر": ("تمام پلیٹ فارمز",
        [("se", "پلیٹ فارم / Platform", "p", ["YouTube", "Facebook", "Instagram", "LinkedIn", "TikTok"]),
         ("tx", "نیش / Niche", "n", "")],
        "social media expert providing content calendar, growth strategy, hashtags, monetization"),
    "📢 ایڈز جنریٹر": ("Google, FB, YT, IG",
        [("tx", "پروڈکٹ / Product", "p", ""),
         ("se", "پلیٹ فارم / Platform", "plt", ["Google", "Facebook", "YouTube", "Instagram", "Email"])],
        "ads expert providing campaign, copy, targeting, budget, creative ideas"),
    "💰 آٹو مونیٹائزیشن": ("کمائی اور پیج",
        [("se", "پلیٹ فارم / Platform", "p", ["YouTube", "Facebook", "Instagram", "TikTok", "Blog"]),
         ("tx", "نیش / Niche", "n", "")],
        "monetization expert providing eligibility, revenue, page creation, tax, legal"),
    "🏢 Google Business Profile": ("GBP اور Local SEO",
        [("tx", "بزنس / Business", "b", ""),
         ("tx", "شہر / City", "c", "")],
        "GBP expert providing setup, verification, photos, posts, reviews, local SEO"),
    # Security & Web
    "🔐 سائبر سیکیورٹی": ("ہیکنگ اور حفاظت",
        [("se", "قسم / Type", "type", ["Security Guide", "Ethical Hacking", "PenTest", "Career"]),
         ("ta", "سوال / Question", "q", "")],
        "cybersecurity expert providing ethical hacking, penetration testing, security frameworks"),
    "🌐 ویب ڈویلپمنٹ": ("HTML, CSS, JS, Python",
        [("se", "ٹیکنالوجی / Tech", "tech", ["HTML+CSS+JS", "React", "Next.js", "Flask", "Django", "Node.js"]),
         ("ta", "کیا بنانا ہے / What to build", "d", "")],
        "web developer providing complete code, responsive design, best practices, deployment"),
    "🎨 گرافک ڈیزائننگ": ("تصویر، لوگو، پوسٹر",
        [("se", "اسٹائل / Style", "s", ["infographic", "realistic", "logo", "cartoon", "3d_avatar"]),
         ("ta", "کیا بنانا ہے / What to create", "d", "")],
        "graphic designer providing detailed design brief, color palette, layout, AI-generated visuals"),
    "🎵 آڈیو + ویڈیو جنریشن": ("اسکرپٹ اور گانے",
        [("tx", "موضوع / Topic", "t", ""),
         ("se", "قسم / Type", "type", ["Video Script", "Song", "Voiceover"])],
        "media expert providing scripts, storyboards, song lyrics, voiceover, music composition"),
    "🧑‍🎤 3D اوتار": ("فیس، ہیئر، کلاتھ",
        [("tx", "تفصیل / Description", "d", ""),
         ("se", "قسم / Type", "type", ["Create Avatar", "Change Face", "Change Hair", "Change Clothes"])],
        "3D artist providing avatar creation, character design, face/hair/cloth transformation"),
    "💻 مکمل کوڈنگ": ("23 زبانیں",
        [("se", "زبان / Language", "lang", ["Python", "Java", "JavaScript", "C", "C++", "C#", "Go", "Rust", "PHP", "Swift", "Kotlin", "Ruby", "TypeScript", "SQL"]),
         ("ta", "کام / Task", "t", "")],
        "senior developer providing complete code, comments, explanation, best practices"),
    "🎮 کارٹون + 3D گیمز": ("گیم ڈویلپمنٹ",
        [("tx", "گیم آئیڈیا / Game Idea", "g", ""),
         ("se", "انجن / Engine", "e", ["Unity", "Unreal", "Godot", "Three.js"]),
         ("se", "پلیٹ فارم / Platform", "p", ["PC", "Mobile", "Web", "Console"])],
        "game designer providing complete GDD, mechanics, levels, art style, sample code"),
}

# ==========================================
# 7. Sidebar
# ==========================================
with st.sidebar:
    st.title("🧠 Master AI OS")
    st.caption("76 شعبے · Bilingual · All-in-One")

    st.markdown("---")
    st.subheader("🎨 فونٹس / Fonts")
    URDU_FONTS = {
        "جمیل نوری نستعلیق": "Jameel Noori Nastaleeq",
        "نوٹو نستعلیق اردو": "Noto Nastaliq Urdu",
        "نوٹو سانس عربی": "Noto Sans Arabic",
        "مہر نستعلیق": "Mehr Nastaleeq",
        "القلم تاج نستعلیق": "AlQalam Taj Nastaleeq",
    }
    ENGLISH_FONTS = {"Inter": "Inter", "Roboto": "Roboto", "Poppins": "Poppins", "Fira Code": "Fira Code"}
    s_uf = st.selectbox("اردو فونٹ:", list(URDU_FONTS.keys()), index=0)
    s_ef = st.selectbox("انگلش فونٹ:", list(ENGLISH_FONTS.keys()), index=0)
    active_u_font = URDU_FONTS[s_uf]
    active_e_font = ENGLISH_FONTS[s_ef]

    st.markdown("---")
    st.subheader("⚙️ ماڈل / Model")
    MODELS = {"GPT-OSS 120B (Best)": "openai/gpt-oss-120b", "GPT-OSS 20B (Fast)": "openai/gpt-oss-20b"}
    sel_model_lbl = st.selectbox("ماڈل:", list(MODELS.keys()), index=0)
    selected_model_id = MODELS[sel_model_lbl]

    drive_service = get_google_drive_service()
    if drive_service:
        st.success("🟢 Drive Connected")
    else:
        st.warning("⚪ Drive متبادل")

    st.markdown("---")
    st.subheader("🎙️ وائس / Voice")
    vg = st.selectbox("زبان / Lang:", list(EDGE_VOICES.keys()), index=0)
    vopts = EDGE_VOICES[vg]
    svl = st.selectbox("آواز:", list(vopts.keys()), index=0)
    selected_voice_id = vopts[svl]
    ssl = st.selectbox("رفتار:", list(VOICE_SPEEDS.keys()), index=2)
    selected_speed_rate = VOICE_SPEEDS[ssl]
    spl = st.selectbox("Pitch:", list(VOICE_PITCHES.keys()), index=1)
    selected_pitch_val = VOICE_PITCHES[spl]

    st.markdown("---")
    st.subheader("🧭 شعبہ / Section")

    SPECIAL = [
        "💬 AI چیٹ (Universal)",
        "📚 ڈرائیو بکس / Drive Books",
        "📸 OCR (تصویر سے متن)",
        "📄 PDF → انفوگرافک",
        "🎬 YouTube ٹرانسکرپٹ",
        "🎧 Whisper آڈیو",
        "🎙️ وائس اسٹوڈیو",
        "🎨 گرافک ڈیزائننگ (Custom)",
    ]
    ALL_SECTIONS = SPECIAL + list(SECTIONS_CONFIG.keys())
    active_section = st.radio("", ALL_SECTIONS, index=0, label_visibility="collapsed")

    st.markdown("---")
    if st.button("🗑️ چیٹ صاف کریں", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "السلام علیکم!"}]
        st.rerun()

# ==========================================
# 8. CSS
# ==========================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&family=Noto+Sans+Arabic:wght@400;600;700&family=Inter:wght@400;600;700&family=Roboto:wght@400;500;700&family=Poppins:wght@400;600&family=Fira+Code:wght@400;500&display=swap');
    @font-face {{ font-family: 'Jameel Noori Nastaleeq'; src: url('https://cdn.jsdelivr.net/gh/UrduFonts/Jameel-Noori-Nastaleeq/Jameel-Noori-Nastaleeq.ttf') format('truetype'); font-display: swap; }}
    @font-face {{ font-family: 'Mehr Nastaleeq'; src: url('https://cdn.jsdelivr.net/gh/UrduFonts/Mehr-Nastaleeq/Mehr-Nastaleeq.ttf') format('truetype'); font-display: swap; }}
    .stApp {{
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #ffffff !important;
        font-family: '{active_u_font}', '{active_e_font}', sans-serif !important;
    }}
    .stChatMessage {{
        border-radius: 16px; padding: 18px 24px; margin-bottom: 18px;
        font-size: 1.15rem !important; line-height: 2 !important;
        color: #ffffff !important;
    }}
    [data-testid="stChatMessage"]:nth-child(odd) {{ background: #1e293b !important; border: 2px solid #38bdf8 !important; }}
    [data-testid="stChatMessage"]:nth-child(even) {{ background: #0f172a !important; border: 1px solid #818cf8 !important; }}
    .stChatMessage p, .stChatMessage div, .stChatMessage span {{
        color: #ffffff !important;
        font-family: '{active_u_font}', '{active_e_font}', sans-serif !important;
    }}
    .stChatInputContainer textarea, .stTextArea textarea, .stTextInput input {{
        font-family: '{active_u_font}', '{active_e_font}', sans-serif !important;
        background-color: #1e293b !important; color: #ffffff !important;
        border-radius: 12px; border: 1px solid #38bdf8; font-size: 1.1rem !important;
    }}
    h1, h2, h3, h4 {{ color: #ffffff !important; font-family: '{active_u_font}', '{active_e_font}', sans-serif !important; }}
    p, div, span, label {{ font-family: '{active_u_font}', '{active_e_font}', sans-serif !important; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 9. Groq Setup
# ==========================================
groq_api_key = get_config("GROQ_API_KEY")
if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY نہیں ملی! Secrets میں ڈالیں۔")
    st.stop()

SYSTEM_PROMPT = """You are Master AI — the world's most versatile BILINGUAL (Urdu + English) multilingual agent.

🌐 LANGUAGE RULE:
- User asks in URDU → Answer in URDU
- User asks in ENGLISH → Answer in English
- Roman Urdu → Answer in Urdu
- Mixed → Answer in both

🎯 RULES:
1. NEVER refuse ANY educational question.
2. Answer in user's language.
3. Comprehensive, professional-grade answers.
4. Include formulas, examples, steps, best practices.
5. Be MORE detailed, not less.
6. Use proper Urdu script (نستعلیق), NOT Roman.
"""

def llm_call(prompt, primary_model=None, temp=0.3):
    if not primary_model:
        primary_model = selected_model_id
    models = [primary_model, "openai/gpt-oss-20b" if "120b" in primary_model else "openai/gpt-oss-120b"]
    for mid in models:
        try:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=mid, temperature=temp)
            r = llm.invoke(prompt)
            c = (r.content or "").strip()
            if c and len(c) > 5:
                return c, mid
        except Exception:
            continue
    return "معذرت، اس وقت جواب نہیں بن سکا۔ دوبارہ کوشش کریں یا سوال آسان کریں۔", primary_model

def voice_widget(text, key_prefix):
    if st.button("🎙️ وائس بنائیں", key=f"{key_prefix}_v"):
        with st.spinner("..."):
            a = create_voice(text[:800], selected_voice_id, selected_speed_rate, selected_pitch_val)
            if a:
                a.seek(0)
                st.audio(a, format="audio/mp3")
                a.seek(0)
                st.download_button("📥 ڈاؤن لوڈ", data=a, file_name=f"{key_prefix}.mp3",
                                   mime="audio/mpeg", key=f"{key_prefix}_dl")

# ==========================================
# 10. Universal Section Renderer
# ==========================================
def render_universal_section(section_name):
    caption, fields, expert_role = SECTIONS_CONFIG[section_name]
    st.header(section_name)
    st.caption(caption)

    user_inputs = {}
    for field in fields:
        ftype, label = field[0], field[1]
        fkey = field[2]
        unique_key = f"{section_name}_{fkey}"

        if ftype == "tx":
            ph = field[3] if len(field) > 3 else ""
            user_inputs[fkey] = st.text_input(label, placeholder=ph, key=unique_key)
        elif ftype == "ta":
            ph = field[3] if len(field) > 3 else ""
            user_inputs[fkey] = st.text_area(label, placeholder=ph, height=120, key=unique_key)
        elif ftype == "se":
            opts = field[3] if len(field) > 3 else []
            user_inputs[fkey] = st.selectbox(label, opts, key=unique_key)

    if st.button("🚀 جواب حاصل کریں", type="primary", key=f"{section_name}_btn"):
        non_empty = {k: v for k, v in user_inputs.items() if v and str(v).strip()}
        if not non_empty:
            st.warning("کم از کم ایک فیلڈ بھریں")
            return

        user_data = "\n".join([f"• {k}: {v}" for k, v in non_empty.items()])
        prompt = f"""{SYSTEM_PROMPT}

Act as an expert {expert_role}.

User Input:
{user_data}

Provide a COMPLETE, PROFESSIONAL answer in the user's language. Include:
1. Comprehensive explanation
2. Step-by-step guide
3. Examples and formulas
4. Best practices
5. Practical tips
6. References
7. Warnings"""

        with st.spinner("تیار کر رہا ہوں..."):
            resp, used = llm_call(prompt)
            st.markdown(resp)
            st.caption(f"⚡ {used}")
            st.download_button("📥 جواب ڈاؤن لوڈ", resp,
                               f"{section_name.replace(' ','_')}.txt", "text/plain",
                               key=f"{section_name}_dl")
            voice_widget(resp, section_name)

# ==========================================
# 11. Main Title
# ==========================================
st.title("🧠 Master AI Multimodal OS")
st.caption(f"76 شعبے · Bilingual · فونٹ: {active_u_font} · ماڈل: {sel_model_lbl}")

# ==========================================
# 12. SPECIAL SECTIONS
# ==========================================
if active_section == "💬 AI چیٹ (Universal)":
    st.header("💬 AI چیٹ — ہر شعبے سے جواب")
    st.caption("اردو یا English میں پوچھیں — گوگل ڈرائیو خودکار تلاش")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant",
            "content": "السلام علیکم! میں Master AI ہوں — 76 شعبوں کا ماہر۔\n\n"
                       "اردو یا English میں کوئی بھی سوال پوچھیں۔"}]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if ui := st.chat_input("اردو یا English میں سوال پوچھیں..."):
        st.session_state.messages.append({"role": "user", "content": ui})
        with st.chat_message("user"):
            st.markdown(ui)

        with st.chat_message("assistant"):
            with st.spinner("🔍 ڈرائیو + عالمی ذرائع..."):
                drive_ctx = ""
                status_msg = "🌍 انٹرنیشنل"
                try:
                    books_content, status = search_drive_books(ui)
                    if status == "found" and books_content:
                        drive_ctx = f"\n\n[📚 Drive Books]:\n{books_content}\n"
                        status_msg = "📚 ڈرائیو سے مواد"
                    elif status == "not_relevant":
                        status_msg = "📚 ڈرائیو میں نہیں → 🌍 انٹرنیشنل"
                except Exception:
                    pass

                full = f"{SYSTEM_PROMPT}{drive_ctx}\n\nUser: {ui}\n\nAssistant:"
                resp, used = llm_call(full)
                st.markdown(resp)
                st.caption(f"⚡ {used} | {status_msg}")

                # 🎨 IMAGE with TRANSLATION FIX
                if has_kw(ui, KEYWORDS_IMAGE):
                    with st.spinner("🎨 تصویر بن رہی ہے (اردو → English ترجمہ)..."):
                        # Translate Urdu to English for image generation
                        translate_prompt = (
                            f"Translate this prompt to a short, clear English description for image generation. "
                            f"Only output the English text, nothing else, no quotes, no explanations. "
                            f"Prompt: {ui}"
                        )
                        eng_prompt, _ = llm_call(translate_prompt, temp=0.1)
                        # Validate translation
                        if (not eng_prompt or "معذرت" in eng_prompt or "Could not" in eng_prompt
                                or len(eng_prompt) > 250 or len(eng_prompt) < 3):
                            eng_prompt = ui

                        img = generate_ai_image(eng_prompt, "infographic") or fallback_image(eng_prompt)
                        if img:
                            img.seek(0)
                            st.image(img, caption=f"Generated: {eng_prompt[:80]}")
                            img.seek(0)
                            st.download_button("📥 تصویر ڈاؤن لوڈ", data=img,
                                               file_name="infographic.png", mime="image/png",
                                               key="c_img_dl")

                # 🎙️ VOICE
                if has_kw(ui, KEYWORDS_VOICE):
                    va = create_voice(resp, selected_voice_id, selected_speed_rate, selected_pitch_val)
                    if va:
                        va.seek(0)
                        st.audio(va, format="audio/mp3")

                st.session_state.messages.append({"role": "assistant", "content": resp})

elif active_section == "📚 ڈرائیو بکس / Drive Books":
    st.header("📚 ڈرائیو بکس + انٹرنیشنل سورسز")
    q = st.text_area("سوال / Question:", height=120, key="dk_q")
    force_intl = st.checkbox("صرف انٹرنیشنل ذرائع", key="dk_intl")

    if st.button("🔍 جواب تلاش کریں", type="primary", key="dk_btn"):
        if not q.strip():
            st.warning("سوال لکھیں")
        else:
            with st.spinner("..."):
                drive_ctx = ""
                source_info = "🌍 انٹرنیشنل ذرائع"
                if not force_intl:
                    bc, status = search_drive_books(q)
                    if status == "found" and bc:
                        drive_ctx = f"\n\n[📚 Drive]:\n{bc}\n"
                        source_info = "📚 گوگل ڈرائیو کتب"
                prompt = f"{SYSTEM_PROMPT}{drive_ctx}\n\nسوال: {q}\n\nمکمل جواب دیں۔"
                resp, used = llm_call(prompt)
                st.markdown(resp)
                st.caption(f"📖 {source_info} | ⚡ {used}")
                st.download_button("📥 ڈاؤن لوڈ", resp, "answer.txt", "text/plain", key="dk_dl")
                voice_widget(resp, "dk")

elif active_section == "📸 OCR (تصویر سے متن)":
    st.header("📸 تصویر سے متن / Image to Text")
    f = st.file_uploader("تصویر:", type=["png", "jpg", "jpeg", "webp"], key="ocr_f")
    extra = st.text_input("اضافی ہدایت:", key="ocr_x")

    if f:
        b = f.read()
        st.image(b, use_container_width=True)
        if st.button("🔍 متن نکالیں", type="primary", key="ocr_btn"):
            with st.spinner("..."):
                r = ocr_image(b, extra)
                st.session_state["ocr_t"] = r
                if not r.startswith("⚠️"):
                    st.success("✅")
                else:
                    st.error(r)

    if "ocr_t" in st.session_state:
        st.text_area("متن:", st.session_state["ocr_t"], height=250, key="ocr_ta")
        st.download_button("📥 ڈاؤن لوڈ", st.session_state["ocr_t"],
                           "ocr.txt", "text/plain", key="ocr_dl")
        voice_widget(st.session_state["ocr_t"], "ocr")

elif active_section == "📄 PDF → انفوگرافک":
    st.header("📄 PDF → انفوگرافک")
    pdf = st.file_uploader("PDF:", type=["pdf"], key="pdf_f")
    mp = st.slider("صفحات:", 1, 50, 10, key="pdf_mp")
    focus = st.text_input("خصوصی توجہ:", key="pdf_focus")

    if pdf:
        b = pdf.read()
        st.info(f"📄 {pdf.name} | {len(b)/1024:.1f} KB")
        if st.button("🔍 متن نکالیں", type="primary", key="pdf_btn"):
            with st.spinner("..."):
                t = extract_pdf_text(b, mp)
                st.session_state["pdf_t"] = t
                st.success(f"✅ {len(t)} حروف")

    if "pdf_t" in st.session_state:
        st.text_area("متن:", st.session_state["pdf_t"][:3000], height=200, key="pdf_prev")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎨 انفوگرافک", key="pdf_inf"):
                with st.spinner("..."):
                    topic = focus or pdf.name
                    tp = f"Translate to English for image: {topic}"
                    eng, _ = llm_call(tp, temp=0.1)
                    if not eng or len(eng) > 250:
                        eng = topic
                    img = generate_ai_image(eng, "infographic") or fallback_image(eng)
                    if img:
                        img.seek(0)
                        st.image(img)
        with c2:
            if st.button("🧠 AI تجزیہ", key="pdf_ai"):
                with st.spinner("..."):
                    p = f"{SYSTEM_PROMPT}\n\nPDF کا تجزیہ:\n\n{st.session_state['pdf_t'][:5000]}"
                    a, _ = llm_call(p)
                    st.session_state["pdf_a"] = a
        if "pdf_a" in st.session_state:
            st.markdown(st.session_state["pdf_a"])
        st.download_button("📥 ڈاؤن لوڈ", st.session_state["pdf_t"],
                           f"{pdf.name}.txt", "text/plain", key="pdf_dl")

elif active_section == "🎬 YouTube ٹرانسکرپٹ":
    st.header("🎬 YouTube ٹرانسکرپٹ")
    url = st.text_input("URL:", key="yt_url")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 ٹرانسکرپٹ", type="primary", key="yt_btn"):
            with st.spinner("..."):
                t, i = get_yt_transcript(url)
                if t:
                    st.session_state["yt_t"] = t
                    st.session_state["yt_i"] = i
                    st.success(f"✅ {i}")
                else:
                    st.error(f"❌ {i}")
    with c2:
        if "yt_t" in st.session_state and st.button("🧠 خلاصہ", key="yt_sum"):
            p = f"{SYSTEM_PROMPT}\n\nخلاصہ:\n\n{st.session_state['yt_t'][:4000]}"
            s, _ = llm_call(p)
            st.session_state["yt_s"] = s

    if "yt_t" in st.session_state:
        st.text_area("ٹرانسکرپٹ:", st.session_state["yt_t"], height=250, key="yt_ta")
        st.download_button("📥", st.session_state["yt_t"], "yt.txt", "text/plain", key="yt_dl")
        voice_widget(st.session_state["yt_t"], "yt")

    if "yt_s" in st.session_state:
        st.markdown(st.session_state["yt_s"])

elif active_section == "🎧 Whisper آڈیو":
    st.header("🎧 Whisper ٹرانسکرپشن")
    up = st.file_uploader("فائل:", type=["mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac", "aac"], key="wh_up")
    wl = st.selectbox("زبان:", ["auto", "ur", "hi", "en", "ar", "pa", "fa"], key="wh_l")

    if up:
        st.audio(up)
        if st.button("📝 ٹرانسکرپٹ", type="primary", key="wh_btn"):
            with st.spinner("..."):
                r = transcribe_whisper(up.read(), up.name, wl)
                st.session_state["wh_t"] = r
                if not r.startswith("⚠️"):
                    st.success("✅")
                else:
                    st.error(r)

    if "wh_t" in st.session_state:
        st.text_area("متن:", st.session_state["wh_t"], height=250, key="wh_ta")
        st.download_button("📥", st.session_state["wh_t"], "w.txt", "text/plain", key="wh_dl")
        voice_widget(st.session_state["wh_t"], "wh")

elif active_section == "🎙️ وائس اسٹوڈیو":
    st.header("🎙️ وائس اسٹوڈیو پرو")
    st.caption(f"آواز: {svl} | زبان: {vg} | رفتار: {ssl}")
    txt = st.text_area("ٹیکسٹ:", height=200, key="vs_t")

    if st.button("🎵 آواز بنائیں", type="primary", key="vs_btn"):
        if not txt.strip():
            st.warning("ٹیکسٹ لکھیں")
        else:
            with st.spinner("..."):
                a = generate_voice_edge(txt, selected_voice_id, selected_speed_rate, selected_pitch_val) or generate_voice_gtts(txt)
                if a:
                    st.session_state["vs_a"] = a.getvalue()
                    st.success("✅")

    if "vs_a" in st.session_state:
        st.audio(st.session_state["vs_a"], format="audio/mp3")
        st.download_button("📥 ڈاؤن لوڈ", st.session_state["vs_a"],
                           "voice.mp3", "audio/mpeg", key="vs_dl")

elif active_section == "🎨 گرافک ڈیزائننگ (Custom)":
    st.header("🎨 گرافک ڈیزائننگ + انفوگرافک")
    st.caption("اردو میں لکھیں — خودکار ترجمہ ہو کر تصویر بنے گی")

    style = st.selectbox("اسٹائل:", ["infographic", "realistic", "logo", "cartoon",
                                       "3d_avatar", "anime", "cyberpunk", "sketch",
                                       "web_design", "game_asset", "engineering"], key="gd_s")
    prompt = st.text_area("کیا بنانا ہے (اردو یا English):", height=120, key="gd_p")
    w = st.slider("چوڑائی:", 512, 2048, 1024, key="gd_w")
    h = st.slider("اونچائی:", 512, 2048, 600, key="gd_h")

    if st.button("🎨 تصویر بنائیں", type="primary", key="gd_btn"):
        if not prompt.strip():
            st.warning("پہلے لکھیں کہ کیا بنانا ہے")
        else:
            with st.spinner("🎨 ترجمہ + تصویر بن رہی ہے..."):
                # Auto-translate
                tp = (f"Translate this to a short English image generation prompt. "
                      f"Output only the English text, nothing else. Prompt: {prompt}")
                eng, _ = llm_call(tp, temp=0.1)
                if not eng or "معذرت" in eng or len(eng) > 250:
                    eng = prompt
                st.info(f"🌐 English: {eng[:200]}")
                img = generate_ai_image(eng, style, w, h) or fallback_image(eng, (w, h))
                if img:
                    img.seek(0)
                    st.image(img)
                    img.seek(0)
                    st.download_button("📥 ڈاؤن لوڈ", img, f"{style}.png", "image/png", key="gd_dl")

# ==========================================
# 13. UNIVERSAL SECTIONS
# ==========================================
elif active_section in SECTIONS_CONFIG:
    render_universal_section(active_section)

# ==========================================
# 14. Footer
# ==========================================
st.markdown("---")
st.caption("🧠 *Master AI Multimodal OS* — 76 شعبے · Bilingual (اردو + English)")
