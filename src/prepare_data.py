import os
import json
from datasets import load_dataset

def build_enterprise_dataset():
    base_dir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # /data exist
    os.makedirs(os.path.join(base_dir,"data"), exist_ok=True)
    output_path = os.path.join(base_dir,"data","dataset_fraud.jsonl")

    print(" ** Descargando dataset bancario desde Hugging Face (Bitext)...")
    # We load a realistic dataset of banking and customer support intents.
    raw_dataset = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", split="train")


    # Mapping intents from the public dataset to our fintech business categories.
    intent_mapping = {
        "dispute_charge": ("aclaracion","medio"),
        "cancel_transfer": ("fraude","alto"),
        "card_linking": ("soporte","bajo"),
        "change_pin": ("soporte","bajo"),
        "check_balance": ("soporte", "bajo"),
        "report_fraud": ("fraude","alto"),
        "lost_card": ("fraude","alto")
    }

    print(" ** Transformando datos al formato estructurado ChatLM JSON ... ")
    processed_cout = 0    
    contadores = {"guardados": 0, "ignorados": 0}

    print(f" ** Procesando y filtrando registros para {output_path}...")

    with open(output_path,"w",encoding="utf-8") as f:
        for row in raw_dataset:
            intent = row["intent"]

            # we only take the data that aligns with our financial business map.
            if intent in intent_mapping:
                categoria, riesgo = intent_mapping[intent]
                texto_usuario = row["instruction"]

                # build concret structure of ChatML JASON
                payload_respuesta = {
                    "categoria": categoria,
                    "riesgo": riesgo    
                }
                structure = {
                    "messages": [
                        {
                        "role": "system",
                        "content": "Analiza el mensaje y responde SOLO con un JSON: {'categoria':'fraude'|'soporte'|'aclaracion', 'riesgo':'alto'|'medio'|'bajo'}"
                        },
                        {"role":"user", "content":texto_usuario},
                        {"role":"assistant", "content": json.dumps(payload_respuesta)}
                    ]
                }
                # write line by line of JSON format    
                f.write(json.dumps(structure, ensure_ascii=False) + "\n")
                contadores["guardados"] += 1
                processed_cout += 1
            else:
                contadores["ignorados"] +=1
    
    print(f" ** ¡Dataset listo...! Se ha generado {processed_cout} ejemplos realistas en: {output_path}")
    print(f" - Guardados (mapeados): {contadores['guardados']}")
    print(f" - Ignorados (no requeridos): {contadores['ignorados']}")

if __name__ == "__main__":
    build_enterprise_dataset()
