
// ==========================================
// NOVA AI - FRONTEND
// Whisper + Ollama + Browser Speech
// ==========================================

const $ = id => document.getElementById(id);

const recordBtn = $("recordBtn");
const recordLabel = $("recordLabel");
const recordIcon = $("recordIcon");
const voiceTitle = $("voiceTitle");
const voiceDescription = $("voiceDescription");
const timer = $("timer");
const waveform = $("waveform");
const voicePanel = document.querySelector(".voice-panel");
const messages = $("messages");
const messageInput = $("messageInput");
const sendBtn = $("sendBtn");
const systemStatus = $("systemStatus");
const connectionStatus = $("connectionStatus");
const muteBtn = $("muteBtn");

let recording = false;
let busy = false;
let muted = false;
let stream = null;
let audioContext = null;
let processor = null;
let source = null;
let analyser = null;
let silentGain = null;
let chunks = [];
let startTime = 0;
let timerInterval = null;
let animationId = null;

const MAX_RECORD_SECONDS = 60;

// ==========================================
// ANIMATED WAVEFORM
// ==========================================

const bars = [];

for (let i = 0; i < 32; i++) {
  const bar = document.createElement("span");
  waveform.appendChild(bar);
  bars.push(bar);
}

function idleWave() {
  bars.forEach((bar, i) => {
    const height = 6 + Math.abs(Math.sin(i * 0.65)) * 12;
    bar.style.height = `${height}px`;
  });
}

idleWave();

function animateWave() {
  if (!recording || !analyser) return;

  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);

  bars.forEach((bar, i) => {
    const index = Math.floor(i * data.length / bars.length);
    const level = data[index] / 255;
    bar.style.height = `${Math.max(6, 6 + level * 47)}px`;
  });

  animationId = requestAnimationFrame(animateWave);
}

// ==========================================
// UI STATE
// ==========================================

function setStatus(text) {
  systemStatus.textContent = `● ${text}`;
}

function setVoiceState(title, description) {
  voiceTitle.textContent = title;
  voiceDescription.textContent = description;
}

function setBusy(value) {
  busy = value;
  recordBtn.disabled = value;
  sendBtn.disabled = value;
  messageInput.disabled = value;
}

function resetRecorder() {
  voicePanel.classList.remove("recording");
  recordBtn.classList.remove("stop");
  recordLabel.textContent = "Start talking";
  recordIcon.textContent = "🎙";
}

function updateTimer() {
  const seconds = Math.floor((Date.now() - startTime) / 1000);

  const min = String(Math.floor(seconds / 60)).padStart(2, "0");
  const sec = String(seconds % 60).padStart(2, "0");

  timer.textContent = `${min}:${sec}`;

  if (seconds >= MAX_RECORD_SECONDS) {
    stopRecording();
  }
}

// ==========================================
// MICROPHONE RECORDING
// ==========================================

recordBtn.addEventListener("click", async () => {
  if (busy) return;

  if (recording) {
    await stopRecording();
  } else {
    await startRecording();
  }
});

async function startRecording() {
  try {
    window.speechSynthesis?.cancel();

    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioContext = new AudioContext();
    await audioContext.resume();

    source = audioContext.createMediaStreamSource(stream);

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;

    // Browser microphone audio capture
    processor = audioContext.createScriptProcessor(4096, 1, 1);

    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;

    chunks = [];

    processor.onaudioprocess = event => {
      if (!recording) return;

      const samples = event.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(samples));
    };

    source.connect(analyser);
    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioContext.destination);

    recording = true;
    startTime = Date.now();

    voicePanel.classList.add("recording");
    recordBtn.classList.add("stop");
    recordLabel.textContent = "Stop & Send";
    recordIcon.textContent = "■";

    setVoiceState("I'm listening...", "Speak naturally into your microphone.");
    setStatus("Recording");

    timer.textContent = "00:00";
    timerInterval = setInterval(updateTimer, 250);

    animateWave();

  } catch (error) {
    await cleanupAudio();
    setVoiceState("Microphone unavailable", error.message);
    setStatus("Microphone error");
  }
}

async function cleanupAudio() {
  clearInterval(timerInterval);
  cancelAnimationFrame(animationId);

  try { processor?.disconnect(); } catch {}
  try { source?.disconnect(); } catch {}
  try { analyser?.disconnect(); } catch {}
  try { silentGain?.disconnect(); } catch {}

  stream?.getTracks().forEach(track => track.stop());

  if (audioContext && audioContext.state !== "closed") {
    await audioContext.close();
  }

  stream = null;
  processor = null;
  source = null;
  analyser = null;
  silentGain = null;
  audioContext = null;
}

async function stopRecording() {
  if (!recording) return;

  recording = false;
  resetRecorder();

  const sampleRate = audioContext.sampleRate;

  await cleanupAudio();
  idleWave();

  setBusy(true);
  setStatus("Processing");
  setVoiceState("Understanding you...", "Whisper is recognising your speech.");

  try {
    const length = chunks.reduce((sum, x) => sum + x.length, 0);

    if (length === 0) {
      throw new Error("No audio was recorded.");
    }

    const samples = new Float32Array(length);
    let offset = 0;

    for (const chunk of chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }

    const converted = resample(samples, sampleRate, 16000);
    const wav = makeWav(converted, 16000);

    const form = new FormData();
    form.append("audio", wav, "speech.wav");

    const response = await fetch("/transcribe", {
      method: "POST",
      body: form
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Speech recognition failed");
    }

    if (!result.text?.trim()) {
      setVoiceState("Didn't catch that", "Try speaking a little closer to your microphone.");
      setStatus("Ready");
      return;
    }

    addMessage("user", result.text);

    await askAI(result.text);

  } catch (error) {
    setVoiceState("Something went wrong", error.message);
    setStatus("Error");
  } finally {
    setBusy(false);
    if (!voicePanel.classList.contains("recording")) {
      recordLabel.textContent = "Start talking";
    }
  }
}

// ==========================================
// AUDIO CONVERSION TO WAV
// ==========================================

function resample(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;

  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const start = i * ratio;
    const end = Math.min(input.length, (i + 1) * ratio);
    const first = Math.floor(start);
    const last = Math.ceil(end);

    let sum = 0;
    let weight = 0;

    for (let j = first; j < last; j++) {
      const overlap = Math.max(
        0,
        Math.min(j + 1, end) - Math.max(j, start)
      );

      if (j < input.length && overlap > 0) {
        sum += input[j] * overlap;
        weight += overlap;
      }
    }

    output[i] = weight ? sum / weight : 0;
  }

  return output;
}

function makeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function write(offset, text) {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  }

  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);

  samples.forEach((sample, i) => {
    const value = Math.max(-1, Math.min(1, sample));

    view.setInt16(
      44 + i * 2,
      value < 0 ? value * 32768 : value * 32767,
      true
    );
  });

  return new Blob([buffer], { type: "audio/wav" });
}

// ==========================================
// CONVERSATION MESSAGES
// ==========================================

function addMessage(role, text) {
  const welcome = $("welcomeMessage");
  if (welcome) welcome.remove();

  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "message-role";
  label.textContent = role === "user" ? "YOU" : "✦ NOVA AI";

  const content = document.createElement("div");
  content.textContent = text;

  bubble.append(label, content);

  if (role === "ai") {
    const actions = document.createElement("div");
    actions.className = "message-actions";

    const playBtn = document.createElement("button");
    playBtn.textContent = "🔊 Play";
    playBtn.onclick = () => speak(text);

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.onclick = () => navigator.clipboard.writeText(text);

    actions.append(playBtn, copyBtn);
    bubble.appendChild(actions);
  }

  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
}

// ==========================================
// OLLAMA AI
// ==========================================

async function askAI(question) {
  setVoiceState("NOVA is thinking...", "Generating an intelligent response.");
  setStatus("Thinking");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question })
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "AI response failed.");
    }

    const answer = result.answer || "I couldn't generate an answer.";

    addMessage("ai", answer);

    setVoiceState("Here's my answer", "Listen or ask another question.");
    setStatus("Ready");

    speak(answer);

  } catch (error) {
    addMessage("ai", `Error: ${error.message}`);
    setVoiceState("AI unavailable", "Check whether Ollama is running.");
    setStatus("Error");
  }
}

// ==========================================
// BROWSER TEXT-TO-SPEECH
// ==========================================

function speak(text) {
  if (muted || !("speechSynthesis" in window)) return;

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-GB";
  utterance.rate = 1;
  utterance.pitch = 1;

  const voices = window.speechSynthesis.getVoices();

  const voice = voices.find(
    v => v.lang === "en-GB" && v.localService
  ) || voices.find(
    v => v.lang.startsWith("en") && v.localService
  ) || voices.find(v => v.lang.startsWith("en"));

  if (voice) utterance.voice = voice;

  utterance.onstart = () => setStatus("Speaking");
  utterance.onend = () => setStatus("Ready");
  utterance.onerror = () => setStatus("Ready");

  window.speechSynthesis.speak(utterance);
}

muteBtn.addEventListener("click", () => {
  muted = !muted;
  muteBtn.textContent = muted ? "🔇" : "🔊";
  muteBtn.title = muted ? "Enable voice replies" : "Mute voice replies";

  if (muted) window.speechSynthesis.cancel();
});

// ==========================================
// TEXT INPUT
// ==========================================

$("messageForm").addEventListener("submit", async event => {
  event.preventDefault();

  if (busy || recording) return;

  const question = messageInput.value.trim();
  if (!question) return;

  messageInput.value = "";
  addMessage("user", question);

  setBusy(true);
  try {
    await askAI(question);
  } finally {
    setBusy(false);
  }
});

// ==========================================
// CLEAR CHAT
// ==========================================

$("clearBtn").addEventListener("click", () => {
  window.speechSynthesis.cancel();

  messages.innerHTML = `
    <div class="welcome" id="welcomeMessage">
      <div class="welcome-icon">✦</div>
      <h3>Your conversation starts here</h3>
      <p>Ask a question using your voice or type a message below.</p>
    </div>
  `;

  setVoiceState("Ready when you are", "Start a new conversation.");
  setStatus("Ready");
  timer.textContent = "00:00";
});

// ==========================================
// SERVER HEALTH CHECK
// ==========================================

async function checkServer() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Backend unavailable");

    connectionStatus.textContent = "● Backend connected";
    connectionStatus.style.color = "#34d399";

  } catch {
    connectionStatus.textContent = "● Backend offline";
    connectionStatus.style.color = "#fb7185";
  }
}

checkServer();
