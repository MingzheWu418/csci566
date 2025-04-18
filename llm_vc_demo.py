import os
import random
import argparse
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference_tts_scale import inference_one_sample
from models import voicecraft
from data.tokenizer import AudioTokenizer, TextTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Combined LLM and VoiceCraft TTS")

    # LLM arguments
    parser.add_argument("--llm_model_path", type=str, default="./my_models/llama8B",
                        help="Path to the LLM model")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                        help="Prompt for the LLM")
    parser.add_argument("--max_length", type=int, default=500,
                        help="Maximum length for LLM generation")

    # VoiceCraft arguments
    parser.add_argument("-m", "--model_name", type=str, default="giga830M",
                        choices=["giga330M", "giga830M", "giga330M_TTSEnhanced", "giga830M_TTSEnhanced"],
                        help="VoiceCraft model to use")
    parser.add_argument("-st", "--silence_tokens", type=int, nargs="*",
                        default=[1388, 1898, 131], help="Silence token IDs")
    parser.add_argument("-casr", "--codec_audio_sr", type=int,
                        default=16000, help="Codec audio sample rate")
    parser.add_argument("-csr", "--codec_sr", type=int, default=50,
                        help="Codec sample rate")
    parser.add_argument("-k", "--top_k", type=float,
                        default=0, help="Top k value")
    parser.add_argument("-p", "--top_p", type=float,
                        default=0.8, help="Top p value")
    parser.add_argument("-t", "--temperature", type=float,
                        default=1, help="Temperature value")
    parser.add_argument("-kv", "--kvcache", type=float, choices=[0, 1],
                        default=0, help="Kvcache value")
    parser.add_argument("-sr", "--stop_repetition", type=int,
                        default=-1, help="Stop repetition for generation")
    parser.add_argument("--sample_batch_size", type=int,
                        default=3, help="Batch size for sampling")
    parser.add_argument("-s", "--seed", type=int,
                        default=1, help="Seed value")
    parser.add_argument("-bs", "--beam_size", type=int, default=50,
                        help="beam size for MFA alignment")
    parser.add_argument("-rbs", "--retry_beam_size", type=int, default=200,
                        help="retry beam size for MFA alignment")
    parser.add_argument("--output_dir", type=str, default="./generated_tts",
                        help="directory to save generated audio")
    parser.add_argument("-oa", "--original_audio", type=str,
                        default="./demo/5895_34622_000026_000002.wav",
                        help="location of audio file")
    parser.add_argument("-ot", "--original_transcript", type=str,
                        default="Gwynplaine had, besides, for his work and for his feats of strength, round his neck and over his shoulders, an esclavine of leather.",
                        help="original transcript")
    parser.add_argument("-co", "--cut_off_sec", type=float, default=3.6,
                        help="cut off point in seconds for input prompt")
    parser.add_argument("-ma", "--margin", type=float, default=0.04,
                        help="margin in seconds between the end of the cutoff words and the start of the next word")
    parser.add_argument("-cuttol", "--cutoff_tolerance", type=float, default=1,
                        help="tolerance in seconds for the cutoff time")

    args = parser.parse_args()
    return args

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def find_closest_word_boundary(alignments, cut_off_sec, margin, cutoff_tolerance=1):
    with open(alignments, 'r') as file:
        # skip header
        next(file)
        cutoff_time = None
        cutoff_index = None
        cutoff_time_best = None
        cutoff_index_best = None
        lines = [l for l in file.readlines()]
        for i, line in enumerate(lines):
            end = float(line.strip().split(',')[1])
            if end >= cut_off_sec and cutoff_time == None:
                cutoff_time = end
                cutoff_index = i
            if end >= cut_off_sec and end < cut_off_sec + cutoff_tolerance and i+1 < len(lines) and float(lines[i+1].strip().split(',')[0]) - end >= margin:
                cutoff_time_best = end + margin * 2 / 3
                cutoff_index_best = i
                break
        if cutoff_time_best != None:
            cutoff_time = cutoff_time_best
            cutoff_index = cutoff_index_best
        return cutoff_time, cutoff_index

def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed for reproducibility
    seed_everything(args.seed)

    print("\n[1/3] Loading LLM model and generating response...")
    # Load LLM model and tokenizer
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)

    # Generate response using LLM
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(inputs["input_ids"], max_length=args.max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"LLM Response to '{args.prompt}':")
    print(f"{generated_text}\n")

    print("[2/3] Loading VoiceCraft model...")
    # Load VoiceCraft model
    voicecraft_name = args.model_name
    if voicecraft_name == "330M":
        voicecraft_name = "giga330M"
    elif voicecraft_name == "830M":
        voicecraft_name = "giga830M"
    elif voicecraft_name == "330M_TTSEnhanced":
        voicecraft_name = "330M_TTSEnhanced"
    elif voicecraft_name == "830M_TTSEnhanced":
        voicecraft_name = "830M_TTSEnhanced"

    model = voicecraft.VoiceCraft.from_pretrained(
        f"pyp1/VoiceCraft_{voicecraft_name.replace('.pth', '')}")
    phn2num = model.args.phn2num
    config = vars(model.args)
    model.to(device)

    # Load audio tokenizer
    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(encodec_fn):
        os.system(
            f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O ./pretrained_models/encodec_4cb2048_giga.th")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)

    # Load text tokenizer
    text_tokenizer = TextTokenizer(backend="espeak")

    # Prepare voice prompt
    print("[3/3] Preparing voice prompt and generating speech...")
    orig_audio = args.original_audio
    orig_transcript = args.original_transcript

    # Move the audio and transcript to temp folder
    temp_folder = "./demo/temp"
    os.makedirs(temp_folder, exist_ok=True)
    os.system(f"cp {orig_audio} {temp_folder}")
    filename = os.path.splitext(orig_audio.split("/")[-1])[0]
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(orig_transcript)

    # Run MFA to get the alignment
    align_temp = f"{temp_folder}/mfa_alignments"
    beam_size = args.beam_size
    retry_beam_size = args.retry_beam_size
    alignments = f"{temp_folder}/mfa_alignments/{filename}.csv"
    if not os.path.isfile(alignments):
        os.system(f"mfa align -v --clean -j 1 --output_format csv {temp_folder} \
                english_us_arpa english_us_arpa {align_temp} --beam {beam_size} --retry_beam {retry_beam_size}")

    # Find cutoff point
    cut_off_sec = args.cut_off_sec
    margin = args.margin
    audio_fn = f"{temp_folder}/{filename}.wav"

    # Make sure cut_off_sec is valid
    info = torchaudio.info(audio_fn)
    audio_dur = info.num_frames / info.sample_rate
    assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"

    cut_off_sec, cut_off_word_idx = find_closest_word_boundary(alignments, cut_off_sec, margin, args.cutoff_tolerance)

    # Use the LLM-generated text as the target transcript
    target_transcript = " ".join(orig_transcript.split(" ")[:cut_off_word_idx+1]) + " " + generated_text
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    # Set up config for voice synthesis
    decode_config = {
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'stop_repetition': args.stop_repetition,
        'kvcache': args.kvcache,
        'codec_audio_sr': args.codec_audio_sr,
        'codec_sr': args.codec_sr,
        'silence_tokens': args.silence_tokens,
        'sample_batch_size': args.sample_batch_size
    }

    # Generate speech for LLM response
    concated_audio, gen_audio = inference_one_sample(
        model,
        argparse.Namespace(**config),
        phn2num,
        text_tokenizer,
        audio_tokenizer,
        audio_fn,
        target_transcript,
        device,
        decode_config,
        prompt_end_frame
    )

    # Save the generated audio
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Format output filenames
    prompt_text = args.prompt.replace(" ", "_")[:20]  # Use first 20 chars of prompt in filename
    seg_save_fn_gen = f"{output_dir}/llm_response_to_{prompt_text}_gen_seed{args.seed}.wav"
    seg_save_fn_concat = f"{output_dir}/llm_response_to_{prompt_text}_concat_seed{args.seed}.wav"

    # Convert to CPU and save
    concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
    torchaudio.save(seg_save_fn_gen, gen_audio, args.codec_audio_sr)
    torchaudio.save(seg_save_fn_concat, concated_audio, args.codec_audio_sr)

    print(f"\nSuccess! Generated audio files saved to:")
    print(f"  - Generated speech: {seg_save_fn_gen}")
    print(f"  - Concatenated original + generated: {seg_save_fn_concat}")

if __name__ == "__main__":
    main()