import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path


def test_inference():

    # PYTHON_CUDA_ALLOC_CONF this prevent PyTorch memory from becoming fragmented into unsuable blocks
    os.environ["PYTHON_CUDA_ALLOC_CONF"]= "expandable_segments:True"

    # 1. Managment relative rutes
    # Set automatic detection where script is, to avoid absoluted broke rutes    
    print(f"....starting process ...")   
    base_dir = Path.cwd()
    adapter_path = base_dir / "modelo_fintech_final"

    #base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #adapter_path = os.path.join(base_dir, "modelo_fintech_final")

    
    # 2. Extreme Quantization Settings (VRAM savings for production)    
    # We load the model in 4-bit (NF4). This reduces memory consumption from ~16GB to just ~5.5GB of VRAM.
    
    print(" ** Cargando modelo base cuantizado y adaptadores LoRA...")
    print(" configurando cuantizacion de 4 bits (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",                          # float16 if GPU is old like 4T
        bnb_4bit_compute_dtype=torch.bfloat16,               # bfloat16 avoids mathematical degradation on modern GPUs
        bnb_4bit_use_double_quant=True,                      # Quantizes the quantization constants to save an additional 0.4 bits per parameter
        llm_int8_enable_fp32_cpu_offload=True
    )

    # 3. Load Model and Tokenizaer
    # It uses a open-source model base on lead industry    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"...** descargando/cargando model base: {model_id}...")

    # token from HugginFace acces read olny
    #hf_token = "HF_TOKEN"
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    except ImportError:
        # excecution out from colab
        import os
        hf_token = os.environ.get("HF_TOKEN")
    
    print(" .... 1.-there is clenan cache before model works....")
    gc.collect()
    tourch.cuda.empty_cache()                 # there is clean cache before model
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"          # avoid atention problems during training model motived for padding left

    # Cargar modelo base
    print("...loading model base ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=bnb_config,                           # Distribuites the layers automatically across the available GPU.
        device_map="auto"
    )

    # Acoplar tus adaptadores entrenados
    print(" ... copling training adapters ... probability set a warnnig ... (PeftModel - get_peft_model )...")    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()                                                  # Configurar en modo evaluacion (desactiva dropout)

    # Simulacion de un correo de un cliente real
    print("... simulating customer email... : ")
    prompt_usuario = "Hola, me urge ayuda. Veo una transferencia que yo no autorice a una cuenta desconocida por $2,000 dolares realizada hace una hora."
    print(f" ... prompt_usuario {prompt_usuario}")

    # Formato estricto ChatML identico al entrenamiento
    print("...ChatML format identical on training...")
    messages = [
        {"role": "system", "content": "Analiza el mensaje y responde SOLO con un JSON: {'categoria': 'fraude'|'soporte'|'aclaracion', 'riesgo': 'alto'|'medio'|'bajo'}"},
        {"role": "user", "content": prompt_usuario}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    print(" ** Generando respuesta estructurada...")
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100, temperature=0.1, do_sample=False)
    
    respuesta = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print("\n ** RESULTADO DEL LLM EN PRODUCCIÓN:...")
    print(respuesta)

if __name__ == "__main__":
    test_inference()