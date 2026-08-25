import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. RUTAS Y CONFIGURACIÓN DEL MODELO

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
adapter_path = os.path.join(base_dir, "modelo_fintech_final")
model_id = "meta-llama/Llama-3-8B-Instruct"

print(" ** Inicializando componentes de IA y cargando pesos...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
# Acoplamos el adaptador entrenado en la Fase 1
print(" ...setting model adapter training ....")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()


# 2. FUNCIoN DE PREDICCIoN PARA EL CHAT
print("...starting Chat prediction function...")
def clasificar_mensaje(mensaje_usuario, historial):
    # Formato de instrucciones identico al entrenamiento
    print("...command format same on training...")
    messages = [
        {
            "role": "system", 
            "content": "Analiza el mensaje y responde SOLO con un JSON: {'categoria': 'fraude'|'soporte'|'aclaracion', 'riesgo': 'alto'|'medio'|'bajo'}"
        },
        {"role": "user", "content": mensaje_usuario}
    ]
    
    # Preparar los tokens para la GPU
    print("...preparing tokens to GPU...")
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    # Generar la respuesta de forma determinista (low temperature)
    print("...generar respuesta de forma determinista ...")
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100, temperature=0.1, do_sample=False)
    
    # Decodificar solo el texto nuevo generado por el modelo
    print("...decoding text on new model generated...")
    respuesta_json = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return respuesta_json

# 3. DISEnO DE LA INTERFAZ DE GRADIO (UI)
print("...starting GUI of GRADIO...")
demo = gr.ChatInterface(
    fn=clasificar_mensaje,
    title=" ** Fintech Risk & Fraud Auditor UI **",
    description="Demo interna para el equipo de operaciones. Introduce el reporte o ticket del cliente para obtener la clasificacion estructurada en tiempo real.",
    examples=[
        "Me clonaron la tarjeta, hay 4 compras pendientes en linea.",
        "¿Como puedo cambiar mi contraseña desde la aplicacion movil?",
        "Tengo un cobro duplicado en mi estado de cuenta de este mes."
    ],
    theme="soft"
)

if __name__ == "__main__":
    # share=True genera el enlace público para abrirlo en cualquier PC o compartirlo con el equipo
    demo.launch(share=True)