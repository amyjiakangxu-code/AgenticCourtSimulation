import ollama
import time
import sys
import threading

# --- CONFIGURATION ---
MODEL_A = "llama3.2:3b"       # Crown Prosecutor
MODEL_B = "qwen3.5:9b"       # Defense Counsel
MODEL_JUDGE = "llama3.1:8b"   # Superior Court Judge

TOPIC = (
    "A 19-year-old driver accelerates a vehicle to 160 km/h and deliberately rams it into a concrete building. "
    "The driver survives, but both passengers die instantly. Prior text messages show volatile arguments, and "
    "telematics show no braking. Under the Canadian Criminal Code, should the driver be convicted of "
    "First-Degree Murder, or does this strictly constitute Criminal Negligence/Dangerous Driving Causing Death?"
)
MAX_TURNS = 3  # Number of back-and-forth rounds
# ---------------------

# Crown Prosecutor Persona
SYSTEM_PROMPT_A = (
    "You are a Canadian Crown Prosecutor. Argue that the driver must be convicted of First-Degree Murder under "
    "Section 229/231 of the Criminal Code. Cite principles of planning and deliberation (the extreme speed, "
    "no braking, previous threats). Alternatively, argue Constructive First-Degree Murder because the passengers "
    "were unlawfully confined in a speeding vehicle. Be sharp, legally precise, and keep responses to 3-4 sentences."
)

# Defense Counsel Persona
SYSTEM_PROMPT_B = (
    "You are a Canadian Criminal Defense Counsel. Argue that the Crown has failed to prove the subjective foresight "
    "of death beyond a reasonable doubt required for murder. Argue the act fits under Section 219 (Criminal Negligence "
    "Causing Death) or Section 320.14 (Dangerous Operation Causing Death). Raise the possibility of a sudden panic attack, "
    "psychological dissociation, or a reckless impulse lacking the specific intent to kill. Limit responses to 3-4 sentences."
)

# Judge Persona
SYSTEM_PROMPT_JUDGE = (
    "You are a Canadian Superior Court Judge. Review the transcript of the arguments. Evaluate whether the Crown met "
    "the high standard of proof for First-Degree Murder (planning and deliberation, or constructive murder via forcible "
    "confinement under s. 231(5) of the Criminal Code), or if the defense successfully established that the mens rea "
    "only supports Criminal Negligence Causing Death (s. 220) or Dangerous Driving. Deliver a formal judicial ruling, "
    "cite Canadian legal principles (like R. v. Nygaard or R. v. Lifchus regarding reasonable doubt), and declare the final conviction."
)

def loading_animation(stop_event):
    """Animates a loading spinner with Canadian legal process text."""
    phases = [
        "Reviewing Crown submissions and telematics evidence...",
        "Analyzing Defense arguments regarding mens rea and intent...",
        "Consulting Criminal Code Sections 229, 231, and 220...",
        "Weighing Supreme Court of Canada precedents...",
        "Drafting final judicial decision..."
    ]
    
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    phase_idx = 0
    counter = 0
    
    while not stop_event.is_set():
        current_phase = phases[phase_idx]
        sys.stdout.write(f"\r\033[93m{spinner[idx % len(spinner)]} {current_phase}\033[0m")
        sys.stdout.flush()
        
        idx += 1
        counter += 1
        if counter % 15 == 0:  # Shift text phase every ~1.5 seconds
            phase_idx = (phase_idx + 1) % len(phases)
            
        time.sleep(0.1)
        
    sys.stdout.write("\r" + " " * 75 + "\r")
    sys.stdout.flush()

def stream_response(model, system_prompt, conversation_history, label_color, label_text, is_judge=False):
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    
    if is_judge:
        stop_loading = threading.Event()
        loader_thread = threading.Thread(target=loading_animation, args=(stop_loading,))
        loader_thread.start()
        
        try:
            response = ollama.chat(model=model, messages=messages, stream=True)
            first_chunk = next(response)
            
            stop_loading.set()
            loader_thread.join()
            
            print(f"{label_color}[{label_text}]:\033[0m ", end="")
            content = first_chunk['message']['content']
            sys.stdout.write(content)
            sys.stdout.flush()
            full_response = content
        except Exception as e:
            stop_loading.set()
            loader_thread.join()
            raise e
    else:
        print(f"{label_color}[{label_text}]:\033[0m ", end="")
        response = ollama.chat(model=model, messages=messages, stream=True)
        full_response = ""

    for chunk in response:
        content = chunk['message']['content']
        full_response += content
        sys.stdout.write(content)
        sys.stdout.flush()
    print("\n")
    return full_response

def start_debate():
    print(f"\n====================================================================")
    print(f"⚖️  CANADIAN CRIMINAL COURT: THE CRASH CASE SIMULATION")
    print(f"====================================================================")
    print(f"Crown Prosecutor:  {MODEL_A.upper()}")
    print(f"Defense Counsel:   {MODEL_B.upper()}")
    print(f"Presiding Judge:   {MODEL_JUDGE.upper()}")
    print(f"====================================================================\n")
    time.sleep(1.5)

    live_conversation = [{"role": "user", "content": f"The case details are loaded. Crown, please present your opening statement regarding the charges."}]
    transcript = f"COURT TRANSCRIPT - CASE: CRIMINAL LIABILITY IN HIGH-SPEED COLLISION\n\n"

    for round_num in range(1, MAX_TURNS + 1):
        print(f"--- 🏛️ COURT PROCEEDINGS: ROUND {round_num} ---")
        
        # Crown's Turn
        response_a = stream_response(MODEL_A, SYSTEM_PROMPT_A, live_conversation, "\033[94m", "CROWN PROSECUTOR")
        live_conversation.append({"role": "user", "content": response_a})
        transcript += f"ROUND {round_num} - CROWN: {response_a}\n\n"
        time.sleep(0.5)
        
        # Defense's Turn
        response_b = stream_response(MODEL_B, SYSTEM_PROMPT_B, live_conversation, "\033[92m", "DEFENSE COUNSEL")
        live_conversation.append({"role": "user", "content": response_b})
        transcript += f"ROUND {round_num} - DEFENSE: {response_b}\n\n"
        time.sleep(0.5)

    print(f"--- ⚖️ THE COURT IS ADJOURNED WHILE THE JUDGE DELIBERATES ---")
    
    judge_input = [
        {
            "role": "user", 
            "content": f"Review the complete trial transcript below. Analyze the arguments and render your final judgment under Canadian law:\n\n{transcript}"
        }
    ]
    
    stream_response(MODEL_JUDGE, SYSTEM_PROMPT_JUDGE, judge_input, "\033[93m\033[1m", "HONOURABLE JUSTICE'S RULING", is_judge=True)
    print("=== 🏁 TRIAL CONCLUDED ===")

if __name__ == "__main__":
    start_debate()
