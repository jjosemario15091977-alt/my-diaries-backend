import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel
from supabase import Client, create_client

# Carrega o .env localizado na mesma pasta deste arquivo
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

# --- CONFIGURAÇÕES DE AMBIENTE ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_KEY = (
    os.getenv("SUPABASE_SERVICE_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or ""
).strip()
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "").strip()
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "").strip()
VAPID_SUBJECT = os.getenv("VAPID_SUBJECT", "mailto:contato@mydiaries.app").strip()
CRON_SECRET = os.getenv("CRON_SECRET", "troque_este_segredo_cron").strip()
TIMEZONE_PADRAO = os.getenv("TIMEZONE", "America/Bahia").strip()

# Inicialização da Aplicação FastAPI
app = FastAPI(title="My Diaries API", version="1.0.0")

# Configuração de CORS (aceita localhost e qualquer deploy/preview da Vercel)
origins_permitidas = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://mydiaries.vercel.app",
    "https://my-diaries-niktdovoq-terapias.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_permitidas,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialização dos Clientes Supabase e Gemini
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"api_version": "v1beta"},
)


# --- MODELOS PYDANTIC ---
class PushSubscriptionIn(BaseModel):
    endpoint: str
    p256dh: str
    auth: str
    device_label: Optional[str] = None


class EventoUpdate(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None


# --- DEPENDÊNCIA DE AUTENTICAÇÃO ---
async def verificar_usuario(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token de autorização ausente ou inválido.")

    token = authorization.split(" ", 1)[1].strip()
    try:
        user_resp = supabase.auth.get_user(token)
        if not user_resp or not user_resp.user:
            raise HTTPException(status_code=401, detail="Usuário não autenticado ou sessão expirada.")
        return {"id": user_resp.user.id, "email": user_resp.user.email}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Erro de autenticação: {str(e)}")


# --- ROTAS DE CONFIGURAÇÃO E PUSH ---
@app.get("/vapid-public-key")
def get_vapid_public_key():
    if not VAPID_PUBLIC_KEY:
        raise HTTPException(status_code=500, detail="VAPID_PUBLIC_KEY não configurada no servidor.")
    return {"public_key": VAPID_PUBLIC_KEY}


@app.post("/push/subscribe")
def salvar_push_subscription(sub: PushSubscriptionIn, user: dict = Depends(verificar_usuario)):
    try:
        dados_upsert = {
            "user_id": user["id"],
            "endpoint": sub.endpoint,
            "p256dh": sub.p256dh,
            "auth": sub.auth,
            "device_label": sub.device_label or "Dispositivo Web",
        }
        supabase.table("push_subscriptions").upsert(dados_upsert, on_conflict="endpoint").execute()
        return {"status": "success", "message": "Inscrição push salva com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar subscrição push: {str(e)}")


# --- ROTAS DE EVENTOS ---
@app.get("/events")
def listar_eventos(user: dict = Depends(verificar_usuario)):
    try:
        resp = (
            supabase.table("events")
            .select("*")
            .eq("user_id", user["id"])
            .order("start_at", desc=False)
            .execute()
        )
        return {"events": resp.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar eventos: {str(e)}")


@app.post("/events/ai")
async def criar_evento_ia(
    texto: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    user: dict = Depends(verificar_usuario),
):
    conteudo_prompt = (texto or "").strip()

    if audio:
        audio_bytes = await audio.read()
        prompt_transcricao = "Transcreva este áudio em português com exatidão."
        try:
            transcr_resp = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=audio.content_type or "audio/webm",
                    ),
                    prompt_transcricao,
                ],
            )
            conteudo_prompt = (transcr_resp.text or "").strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro na transcrição do áudio: {str(e)}")

    if not conteudo_prompt:
        raise HTTPException(status_code=400, detail="Nenhum texto ou áudio fornecido para agendamento.")

    try:
        tz = ZoneInfo(TIMEZONE_PADRAO)
    except Exception:
        tz = ZoneInfo("America/Bahia")
    agora_iso = datetime.now(tz).isoformat()

    prompt_ia = f"""
    Data e hora atual de referência ({TIMEZONE_PADRAO}): {agora_iso}

    Analise o texto abaixo e extraia os detalhes de agendamento em formato JSON estrito, contendo exatamente estas chaves:
    - "title": Título claro do compromisso.
    - "category": Apenas "Pessoal" ou "Profissional".
    - "start_at": Data e hora de início no formato ISO 8601 (AAAA-MM-DDTHH:MM:SS), considerando o fuso horário {TIMEZONE_PADRAO}.
    - "end_at": Data e hora de término no formato ISO 8601 (se não especificado, defina 1 hora após o início).
    - "location": Local do evento (endereço ou nome do estabelecimento, ou string vazia se não houver).
    - "description": Breve descrição adicional (ou string vazia).

    Texto do usuário: "{conteudo_prompt}"
    """

    try:
        ia_resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_ia,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        dados_evento = json.loads(ia_resp.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao interpretar compromisso com a IA: {str(e)}")

    try:
        novo_registro = {
            "user_id": user["id"],
            "title": dados_evento.get("title", "Compromisso"),
            "category": dados_evento.get("category", "Pessoal"),
            "start_at": dados_evento.get("start_at"),
            "end_at": dados_evento.get("end_at"),
            "location": dados_evento.get("location", ""),
            "description": dados_evento.get("description", ""),
            "transcription": conteudo_prompt,
        }
        insert_resp = supabase.table("events").insert(novo_registro).execute()
        evento_criado = insert_resp.data[0] if insert_resp.data else novo_registro
        return {"status": "success", "event": evento_criado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar evento no banco de dados: {str(e)}")


@app.delete("/events/{evento_id}")
def excluir_evento(evento_id: str, user: dict = Depends(verificar_usuario)):
    try:
        supabase.table("events").delete().eq("id", evento_id).eq("user_id", user["id"]).execute()
        return {"status": "success", "message": "Evento excluído com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao excluir evento: {str(e)}")


@app.delete("/account")
def excluir_conta(user: dict = Depends(verificar_usuario)):
    try:
        supabase.table("events").delete().eq("user_id", user["id"]).execute()
        supabase.table("push_subscriptions").delete().eq("user_id", user["id"]).execute()
        supabase.auth.admin.delete_user(user["id"])
        return {"status": "success", "message": "Conta e dados removidos permanentemente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar exclusão LGPD da conta: {str(e)}")