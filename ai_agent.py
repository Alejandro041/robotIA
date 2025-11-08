import os, json


# ----- Ajusta a tu proveedor preferido -----
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))


PROMPT_TEMPLATE = """
Eres el cerebro de un robot de almacén compacto con ruedas, LIDAR y bandeja frontal.
Decide UNA sola acción entre: ["seguir", "reducir_velocidad", "replanificar", "pausa_mantenimiento"].


Contexto (JSON):
{context_json}


Criterios:
- Si el LIDAR mínimo < 0.8 => evitar colisión: preferir "reducir_velocidad" o "replanificar" si el camino está bloqueado.
- Si el z-score de corriente > 2.5 => "pausa_mantenimiento".
- Si faltan menos de 10 puntos de ruta => "seguir".


Responde SOLO un JSON con la clave "accion".
"""


# -----------------------
# Fallback determinista
# -----------------------
def rule_fallback(context):
    lidar_min = context.get("lidar_min", 9.9)
    blocked = context.get("path_blocked", False)
    zscore = context.get("current_z", 0.0)
    path_left = context.get("path_len_left", 999)

    if zscore > 2.5:
        return {"accion": "pausa_mantenimiento"}
    if blocked:
        return {"accion": "replanificar"}
    if lidar_min < 0.8:
        return {"accion": "reducir_velocidad"}
    if path_left < 10:
        return {"accion": "seguir"}
    return {"accion": "seguir"}


# -----------------------
# OpenAI (opcional)
# -----------------------
def openai_decide(context):
    from openai import OpenAI
    client = OpenAI()

    context_json = json.dumps(context, ensure_ascii=False)
    prompt = PROMPT_TEMPLATE.format(context_json=context_json)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    txt = resp.choices[0].message.content
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and "accion" in data:
            return data
    except Exception:
        pass

    return rule_fallback(context)


# API pública
def decide_action(context: dict) -> dict:
    """
    Devuelve {"accion": "seguir|reducir_velocidad|replanificar|pausa_mantenimiento"}.
    Funciona offline con reglas; si hay OPENAI_API_KEY, intenta LLM.
    """
    if USE_OPENAI:
        try:
            return openai_decide(context)
        except Exception:
            return rule_fallback(context)

    return rule_fallback(context)
