// DOM elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const handCountDiv = document.getElementById('handCount');
const faceCountDiv = document.getElementById('faceCount');
const movementCountDiv = document.getElementById('movementCount');
const gestureCountDiv = document.getElementById('gestureCount');
const confidenceValueDiv = document.getElementById('confidenceValue');
const fpsCounter = document.getElementById('fpsCounter');
const systemLogs = document.getElementById('systemLogs');
const gestureName = document.getElementById('gestureName');
const devInfoBtn = document.getElementById('devInfoBtn');
const devCard = document.getElementById('devCard');
const overlay = document.getElementById('overlay');
const closeBtn = document.getElementById('closeBtn');
const ctx = canvas.getContext('2d');

// State variables
let isDetecting = false;
let handModel = null;
let faceModel = null;
let animationFrameId = null;
let lastUpdate = 0;
let frameCount = 0;
let fps = 0;
let movementCount = 0;
let gestureCount = 0;
let lastGesture = "None";
let lastGestureTime = 0;

// Set canvas dimensions to match video
function setupCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// Access webcam
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        video.srcObject = stream;

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                setupCanvas();
                resolve();
            };
        });
    } catch (err) {
        statusDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Error accessing webcam: ${err.message}`;
        statusDiv.className = "status error";
        logMessage(`ERROR: Webcam access denied - ${err.message}`);
        console.error("Error accessing webcam:", err);
        return Promise.reject(err);
    }
}

// Load ML models
async function loadModels() {
    statusDiv.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Loading hand detection model...`;
    statusDiv.className = "status loading";
    logMessage("Loading hand detection model...");

    try {
        handModel = await handpose.load();

        statusDiv.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Loading face detection model...`;
        logMessage("Loading face detection model...");
        faceModel = await blazeface.load();

        statusDiv.innerHTML = `<i class="fas fa-check-circle"></i> Models loaded successfully!`;
        statusDiv.className = "status success";
        logMessage("Models loaded successfully!");

        return true;
    } catch (err) {
        statusDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Error loading models: ${err.message}`;
        statusDiv.className = "status error";
        logMessage(`ERROR: Model loading failed - ${err.message}`);
        console.error("Error loading models:", err);
        return false;
    }
}

// Determine handedness (left/right) based on landmark positions
function determineHandedness(landmarks) {
    // Calculate center of hand
    let sumX = 0;
    for (let i = 0; i < landmarks.length; i++) {
        sumX += landmarks[i][0];
    }
    const avgX = sumX / landmarks.length;

    // Calculate wrist position relative to fingers
    const wristX = landmarks[0][0];
    let fingerAvgX = 0;
    for (let i = 1; i < landmarks.length; i++) {
        fingerAvgX += landmarks[i][0];
    }
    fingerAvgX /= (landmarks.length - 1);

    // Determine handedness based on wrist position relative to fingers
    return fingerAvgX < wristX ? "right" : "left";
}

// Recognize gestures based on hand landmarks
function recognizeGesture(landmarks) {
    // Thumb positions
    const thumbTip = landmarks[4];
    const thumbIp = landmarks[3];
    const thumbMcp = landmarks[2];

    // Index finger positions
    const indexTip = landmarks[8];
    const indexDip = landmarks[7];
    const indexPip = landmarks[6];
    const indexMcp = landmarks[5];

    // Middle finger positions
    const middleTip = landmarks[12];
    const middleDip = landmarks[11];
    const middlePip = landmarks[10];
    const middleMcp = landmarks[9];

    // Ring finger positions
    const ringTip = landmarks[16];
    const ringDip = landmarks[15];
    const ringPip = landmarks[14];
    const ringMcp = landmarks[13];

    // Pinky positions
    const pinkyTip = landmarks[20];
    const pinkyDip = landmarks[19];
    const pinkyPip = landmarks[18];
    const pinkyMcp = landmarks[17];

    // Palm base (wrist)
    const wrist = landmarks[0];

    // Calculate distances
    const thumbIndexDist = Math.hypot(
        thumbTip[0] - indexTip[0],
        thumbTip[1] - indexTip[1]
    );

    const thumbMiddleDist = Math.hypot(
        thumbTip[0] - middleTip[0],
        thumbTip[1] - middleTip[1]
    );

    // Thumbs up detection
    if (thumbTip[1] < thumbIp[1] && thumbTip[1] < thumbMcp[1] &&
        indexTip[1] > indexDip[1] && middleTip[1] > middleDip[1] &&
        ringTip[1] > ringDip[1] && pinkyTip[1] > pinkyDip[1]) {
        return "Thumbs Up 👍";
    }

    // Peace sign (victory)
    if (indexTip[1] < indexDip[1] && middleTip[1] < middleDip[1] &&
        ringTip[1] > ringDip[1] && pinkyTip[1] > pinkyDip[1] &&
        thumbTip[1] > thumbIp[1]) {
        return "Peace ✌️";
    }

    // OK sign
    if (thumbIndexDist < 40 &&
        thumbTip[1] < thumbIp[1] &&
        middleTip[1] > middleDip[1] &&
        ringTip[1] > ringDip[1] &&
        pinkyTip[1] > pinkyDip[1]) {
        return "OK 👌";
    }

    // Fist
    if (thumbTip[1] > thumbIp[1] &&
        indexTip[1] > indexDip[1] &&
        middleTip[1] > middleDip[1] &&
        ringTip[1] > ringDip[1] &&
        pinkyTip[1] > pinkyDip[1]) {
        return "Fist 👊";
    }

    // Pointing
    if (indexTip[1] < indexDip[1] &&
        middleTip[1] > middleDip[1] &&
        ringTip[1] > ringDip[1] &&
        pinkyTip[1] > pinkyDip[1]) {
        return "Pointing 👉";
    }

    // Open hand
    if (thumbTip[1] < thumbIp[1] &&
        indexTip[1] < indexDip[1] &&
        middleTip[1] < middleDip[1] &&
        ringTip[1] < ringDip[1] &&
        pinkyTip[1] < pinkyDip[1]) {
        return "Open Hand 🖐️";
    }

    // Rock on
    if (indexTip[1] < indexDip[1] &&
        pinkyTip[1] < pinkyDip[1] &&
        thumbTip[1] < thumbIp[1] &&
        middleTip[1] > middleDip[1] &&
        ringTip[1] > ringDip[1]) {
        return "Rock On 🤘";
    }

    // Love
    if (thumbTip[1] > thumbIp[1] &&
        indexTip[1] < indexDip[1] &&
        thumbIndexDist < 40) {
        return "Love ❤️";
    }

    return "Unknown";
}

// Draw detected hands with improved accuracy
function drawHands(predictions) {
    handCountDiv.textContent = predictions.length;

    for (let i = 0; i < predictions.length; i++) {
        const prediction = predictions[i];
        const landmarks = prediction.landmarks;

        // Determine handedness
        const handedness = determineHandedness(landmarks);
        const handColor = handedness === "left" ? "var(--hand-left)" : "var(--hand-right)";

        // Draw hand landmarks
        for (let j = 0; j < landmarks.length; j++) {
            const landmark = landmarks[j];

            ctx.beginPath();
            ctx.arc(landmark[0], landmark[1], 6, 0, 2 * Math.PI);
            ctx.fillStyle = handColor;
            ctx.fill();
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Draw finger connections
        const fingers = {
            thumb: [0, 1, 2, 3, 4],
            indexFinger: [0, 5, 6, 7, 8],
            middleFinger: [0, 9, 10, 11, 12],
            ringFinger: [0, 13, 14, 15, 16],
            pinky: [0, 17, 18, 19, 20]
        };

        ctx.strokeStyle = handColor;
        ctx.lineWidth = 3;

        for (const finger in fingers) {
            const points = fingers[finger];
            ctx.beginPath();

            for (let k = 0; k < points.length; k++) {
                const point = landmarks[points[k]];
                if (k === 0) {
                    ctx.moveTo(point[0], point[1]);
                } else {
                    ctx.lineTo(point[0], point[1]);
                }
            }

            ctx.stroke();
        }

        // Draw hand label
        ctx.fillStyle = handColor;
        ctx.font = "bold 16px 'Consolas', monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${handedness.toUpperCase()} HAND`, landmarks[0][0], landmarks[0][1] - 25);

        // Recognize gesture
        const gesture = recognizeGesture(landmarks);
        if (gesture !== "Unknown") {
            ctx.fillStyle = "var(--success)";
            ctx.font = "bold 18px 'Consolas', monospace";
            ctx.fillText(gesture, landmarks[0][0], landmarks[0][1] + 40);

            // Update gesture display if it's a new gesture
            if (gesture !== lastGesture || Date.now() - lastGestureTime > 2000) {
                gestureName.textContent = gesture;
                lastGesture = gesture;
                lastGestureTime = Date.now();

                // Count as movement
                movementCount++;
                movementCountDiv.textContent = movementCount;

                // Count as gesture
                gestureCount++;
                gestureCountDiv.textContent = gestureCount;

                // Log the gesture
                logMessage(`Gesture detected: ${gesture}`);
            }
        }
    }
}

// Draw detected faces with square boxes
function drawFaces(predictions) {
    faceCountDiv.textContent = predictions.length;

    for (let i = 0; i < predictions.length; i++) {
        const prediction = predictions[i];
        const start = prediction.topLeft;
        const end = prediction.bottomRight;
        const size = [end[0] - start[0], end[1] - start[1]];

        // Draw face bounding box as a square
        const maxDim = Math.max(size[0], size[1]);
        const centerX = (start[0] + end[0]) / 2;
        const centerY = (start[1] + end[1]) / 2;
        const squareStartX = centerX - maxDim/2;
        const squareStartY = centerY - maxDim/2;

        ctx.strokeStyle = "var(--face-color)";
        ctx.lineWidth = 3;
        ctx.strokeRect(squareStartX, squareStartY, maxDim, maxDim);

        // Draw face label
        ctx.fillStyle = "var(--face-color)";
        ctx.font = "bold 16px 'Consolas', monospace";
        ctx.textAlign = "center";
        ctx.fillText("FACE DETECTED", centerX, squareStartY - 10);

        // Draw face landmarks
        const landmarks = prediction.landmarks;

        if (landmarks) {
            for (let j = 0; j < landmarks.length; j++) {
                const landmark = landmarks[j];

                ctx.beginPath();
                ctx.arc(landmark[0], landmark[1], 4, 0, 2 * Math.PI);
                ctx.fillStyle = "var(--face-color)";
                ctx.fill();
            }
        }
    }
}

// Calculate FPS
function calculateFPS() {
    frameCount++;
    const now = performance.now();
    const elapsed = now - lastUpdate;

    if (elapsed >= 1000) {
        fps = Math.round((frameCount * 1000) / elapsed);
        frameCount = 0;
        lastUpdate = now;
        fpsCounter.textContent = fps;

        // Update confidence based on FPS
        const confidence = Math.min(100, Math.round((fps / 30) * 100));
        confidenceValueDiv.textContent = `${confidence}%`;
    }
}

// Log system messages
function logMessage(message) {
    const now = new Date();
    const timeString = `[${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')} ${now.getHours() >= 12 ? 'PM' : 'AM'}]`;

    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `<span class="log-time">${timeString}</span> ${message}`;

    systemLogs.appendChild(logEntry);
    systemLogs.scrollTop = systemLogs.scrollHeight;
}

// Main detection function
async function detect() {
    if (!isDetecting) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        // Detect hands
        const handPredictions = await handModel.estimateHands(video, {
            flipHorizontal: false,  // Corrected mirroring
            maxHands: 2
        });

        // Detect faces
        const facePredictions = await faceModel.estimateFaces(video, {
            flipHorizontal: false  // Corrected mirroring
        });

        // Draw results
        if (handPredictions.length > 0) {
            drawHands(handPredictions);
        }

        if (facePredictions.length > 0) {
            drawFaces(facePredictions);
        }

        // Update FPS
        calculateFPS();
    } catch (err) {
        console.error("Detection error:", err);
        logMessage(`ERROR: Detection failed - ${err.message}`);
    }

    animationFrameId = requestAnimationFrame(detect);
}

// Start detection
async function startDetection() {
    if (isDetecting) return;

    try {
        await setupCamera();
        statusDiv.innerHTML = `<i class="fas fa-camera"></i> Setting up camera...`;
        logMessage("Initializing camera...");

        if (!handModel || !faceModel) {
            const modelsLoaded = await loadModels();
            if (!modelsLoaded) return;
        }

        isDetecting = true;
        statusDiv.innerHTML = `<i class="fas fa-check-circle"></i> Detection running...`;
        statusDiv.className = "status success";
        lastUpdate = performance.now();
        frameCount = 0;
        logMessage("Detection started");
        detect();
    } catch (err) {
        statusDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Error starting detection: ${err.message}`;
        statusDiv.className = "status error";
        logMessage(`ERROR: Detection start failed - ${err.message}`);
        console.error("Error starting detection:", err);
    }
}

// Stop detection
function stopDetection() {
    isDetecting = false;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    statusDiv.innerHTML = `<i class="fas fa-pause-circle"></i> Detection stopped.`;
    statusDiv.className = "status";
    handCountDiv.textContent = "0";
    faceCountDiv.textContent = "0";
    fpsCounter.textContent = "0";
    movementCountDiv.textContent = "0";
    gestureCountDiv.textContent = "0";
    confidenceValueDiv.textContent = "0%";
    gestureName.textContent = "None";
    logMessage("Detection stopped");
}

// Toggle developer info
function toggleDevInfo() {
    devCard.style.display = devCard.style.display === 'block' ? 'none' : 'block';
    overlay.style.display = overlay.style.display === 'block' ? 'none' : 'block';
}

// Event listeners
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
devInfoBtn.addEventListener('click', toggleDevInfo);
closeBtn.addEventListener('click', toggleDevInfo);
overlay.addEventListener('click', toggleDevInfo);

// Initialize models on load
window.addEventListener('load', () => {
    statusDiv.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Loading machine learning models...`;
    statusDiv.className = "status loading";
    logMessage("Loading machine learning models...");
    loadModels();
});

// Handle window resize
window.addEventListener('resize', setupCanvas);
