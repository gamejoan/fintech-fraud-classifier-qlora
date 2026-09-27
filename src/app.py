import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. RUTAS Y CONFIGURACIÓN DEL MODELO

# PYTORCH_CUDA_ALLOC_CONF this prevent PyTorch memory from becoming --fragmented into unusable blocks
os.environÑ["PYTORCH_CUDA_ALLOC_CONF"]= "expandable_segments:True"

# 1. Managment relative rutes
# Set automatic detection where script is, to avoid absoluted broke rutes
print(f".....starting process ....")
base_dir = Path.cwd()

#base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#adapter_path = os.path.join(base_dir, "modelo_fintech_final")

# 2. Extreme Quantization Settings (VRAM savings for production)
# We load the model in 4-bit (NF4). This reduces memory consumtion from 16GB to just 5.5GB of VRAM
print(" ** Loading model base quantization and LoRA adapters ....")
print(" ** Inicializando componentes de IA y cargando pesos...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                         # float16 if GPU is old like 4T
    bnb_4bit_compute_dtype=torch.bfloat16,              # bfloat16 avoids mathematial degradation on modern GPUs
    bnb_4bit_use_double_quant=True,                     # Quantizes the quantization constants to save an additional 0.4 bits per parameter
    llm_int8_enable_fp32_cpu_offload=True
)

# 3. Load Model and Tokenizaer
# It uses a open-source model base on lead industry
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"... descargando/cargando model base: {model_id}...")

# 3.1 Token from HugginFace acces read only
#hf_token = "HF_TOKEN"
hf_token = None                  # Initialize to avoid UnboundLocalError
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")      
except ImportError:
    # excecution out from colab
    import os
    hf_token = os.environ.get("HF_TOKEN")

print(".... 1.- ther is clean a cache before model works....")
gc.collect()
torch.cuda.empty_cache()                  # ther is clean cache before model


tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"                # avoid atention problems during training model motived for padding left

# Loading model base
print("...loading model base ...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    quantization_config=bnb_config,                 # Distribuites the layers automatically across the available GPU
    device_map="auto"
)

# Acoplamos el adaptador entrenado en la Fase 1
print(" ...setting model adapter training ....")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()                                           # Configurar en modo evaluacion (desactiva dropout)


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
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=False)
    
    # Decodificar solo el texto nuevo generado por el modelo
    print("...decoding text on new model generated...")
    respuesta_json = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
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
    # share=True genera el enlace publico para abrirlo en cualquier PC o compartirlo con el equipo
    demo.launch(share=True)