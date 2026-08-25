import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

def test_inference():

    base_dir = Path.cwd()
    adapter_path = base_dir / "modelo_fintech_final"

    #base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #adapter_path = os.path.join(base_dir, "modelo_fintech_final")

    model_id = "meta-llama/Llama-3-8B-Instruct"

    print(" ** Cargando modelo base cuantizado y adaptadores LoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Cargar modelo base
    print("...loading model base ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Acoplar tus adaptadores entrenados
    print(" ... copling training adapters ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval() # Configurar en modo evaluacion (desactiva dropout)

    # Simulacion de un correo de un cliente real
    print("... simulating customer email...")
    prompt_usuario = "Hola, me urge ayuda. Veo una transferencia que yo no autorice a una cuenta desconocida por $2,000 dolares realizada hace una hora."
    
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
    print("\n ** RESULTADO DEL LLM EN PRODUCCIÓN:")
    print(respuesta)

if __name__ == "__main__":
    test_inference()