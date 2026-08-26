import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from pathlib import Path

def run_training():
    import os    
    #  PYTORCH_CUDA_ALLOC_CONF this prevents PyTorch memory from becoming fragmented into unusable blocks 
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 1. Managment relative rutes
    # Set automatic detection where script is, to avoid absoluted broke rutes
    print(f" ... starting process...")   
    base_dir = Path.cwd()
    data_path = base_dir / "data" / "dataset_fraude.jsonl"
    output_dir = base_dir / "modelo_fintech_final"

    #base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # En Colab, tu ruta raiz es el directorio del repositorio clonado
    #base_dir = "/content/fintech-fraud-classifier-qlora" 
    #data_path = os.path.join(base_dir, "data", "dataset_fraude.jsonl")
    #output_dir = os.path.join(base_dir, "modelo_fintech_final")

    print(f" * Cargando dataset desde: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoudError(f" X Error: No se encontro el archivo de datos en la ruta {data_path}. Asegurate de haberlo creado.")


    # 2. Extreme Quantization Settings (VRAM savings for production)    
    # We load the model in 4-bit (NF4). This reduces memory consumption from ~16GB to just ~5.5GB of VRAM.
    print(" * Configurando cuantizacion de 4 bits (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # float16 if GPU is old like 4T  
        bnb_4bit_compute_dtype=torch.float16,  # bfloat16 avoids mathematical degradation on modern GPUs                                               
        bnb_4bit_use_double_quant=True,         # Quantizes the quantization constants to save an additional 0.4 bits per parameter
        llm_int8_enable_fp32_cpu_offload=True
    )

    # 3. Load Model and Tokenizer
    # It uses a ope-source model base on lead industry
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
    print(f" ** Descargando/Cargando modelo base: {model_id}...")

    # token from HugginFace acces read only
    #hf_token = "HF_TOKEN "
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    except ImportError:
        # excecution out from Colab
        import os
        hf_token = os.environ.get("HF_TOKEN")
    
    print(" ...1-there is clean cache before model works...")
    gc.collect()
    torch.cuda.empty_cache()          # there is clean cache before model 

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"            # avoid atention problems during training model motived for padding left

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto"                       # Distributes the layers automatically across the available GPU.    
    )

    # 4. Already set Model to PEFT\LoRA
    # We freeze 99% of the model and prepare the layers to receive the low-degradation adapters.
    print(" ** Aplicamos adaptadores LoRA (Low-Rank Adaptation)...")
    # there is clean cache before model 
    print(" ...2-there is clean cache before model works...")
    gc.collect()
    torch.cuda.empty_cache()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,                  # Matrix range. 16 offers an excellent balance between precision and speed.
        lora_alpha=32,         # Scalling factor. Typically twice the value of r
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # it show on cmd real percentage on training process (-1%)

    # 5. Load local DataSet 
    # data_path must be passed on str path in order to iterate data_path files
    dataset = load_dataset("json", data_files=str(data_path))

    # 6. Training HiperParameters (aligned industry standars)
    print(" ** Configurando hiperparametros de entrenamiento ....")
    training_args = SFTConfig(   #TrainingArguments(
        output_dir = os.path.join(base_dir, "checkpoints"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        optim="paged_adamw_8bit",
        save_strategy="no",
        report_to="none",
        dataset_text_field="messages",
        max_seq_length=512
    ) 

    # 7. Start Training ...
    print("...starting Training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        #dataset_text_field="messages",      # Specifies that the structured message list format (ChatML) will be used.
        #max_seq_length=512,                 # Truncates long texts to protect video memory.
        tokenizer=tokenizer,
        args=training_args
    )

    # 8. Execution
    print(" ** !Arrancando el proceso de Fine-Tuning en la GPU!...")
    trainer.train()

    # 9. Save the final Adaptation
    # We save the trained weights. This folder will be very small in size and can be attached to the original model in seconds.
    print(f"** Guardando adaptadores entrenados en: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    print(" ** !Proceso finalizado con exito! El modelo esta listo para produccion.")

    if __name__ == "__main__":
        run_training()
    
