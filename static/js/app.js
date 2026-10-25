/**
 * Core Application Frontend State Controller script for the Enterprise IT Support Copilot.
 */

function handleKeyPress(event) {
    if (event.key === "Enter") {
        submitUserMessage();
    }
}

async function submitUserMessage() {
    const inputField = document.getElementById("userInput");
    const sendButton = document.getElementById("sendBtn");
    const messageContainer = document.getElementById("chatMessages");
    const questionText = inputField.value.trim();

    if (!questionText) return;

    // 1. Lock UI controllers to avoid collision during active runtime pipelines
    inputField.value = "";
    inputField.disabled = true;
    sendButton.disabled = true;

    // Apply User Message Component Node to Dashboard Container Layout
    appendMessageBalloon("user", "You", questionText);
    messageContainer.scrollTop = messageContainer.scrollHeight;

    // 2. Append Loading Placeholder Component Node
    const loadingId = "load-" + Date.now();
    appendMessageBalloon("assistant", "AI", "Processing query vectors across local graph circuits...", loadingId);
    messageContainer.scrollTop = messageContainer.scrollHeight;

    try {
        // 3. Disconnect and call the production FastAPI Endpoint matrix parameters
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: questionText })
        });

        if (!response.ok) {
            throw new Error(`HTTP Error Status Exception: ${response.status}`);
        }

        const data = await response.json();
        
        // Remove placeholder and parse actual data outputs
        document.getElementById(loadingId).remove();

        // 4. Construct response metrics block and append payload points
        appendMessageWithMetrics(data);
        messageContainer.scrollTop = messageContainer.scrollHeight;

    } catch (error) {
        console.error("Pipeline failure:", error);
        document.getElementById(loadingId).remove();
        appendMessageBalloon("assistant", "AI", `System Communication Error: Failed to resolve query. Details: ${error.message}`);
    } finally {
        // Restore active user element controllers
        inputField.disabled = false;
        sendButton.disabled = false;
        inputField.focus();
    }
}

function appendMessageBalloon(role, senderName, textContent, optionalId = null) {
    const container = document.getElementById("chatMessages");
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;
    if (optionalId) wrapper.id = optionalId;

    wrapper.innerHTML = `
        <div class="avatar">${role === "user" ? "U" : "AI"}</div>
        <div class="bubble">${textContent}</div>
    `;
    container.appendChild(wrapper);
}

function appendMessageWithMetrics(payload) {
    const container = document.getElementById("chatMessages");
    const wrapper = document.createElement("div");
    wrapper.className = "message assistant";

    // Resolve customized provenance tracking CSS badges based on actual source metrics
    let sourceClass = "src-direct";
    let sourceLabel = "Direct Response Engine";
    if (payload.source_used === "kb") {
        sourceClass = "src-kb";
        sourceLabel = "Secure Corporate Data Systems";
    } else if (payload.source_used === "web_search") {
        sourceClass = "src-web";
        sourceLabel = "Internet Search is off" //"Live Tavily Public Network Match";
    }

    wrapper.innerHTML = `
        <div class="avatar">AI</div>
        <div>
            <div class="bubble">${payload.answer}</div>
            <div class="meta-metrics">
                <span class="badge ${sourceClass}"><i class="fa-solid fa-database"></i> Source: ${sourceLabel}</span>
                <span class="badge"><i class="fa-solid fa-bolt"></i> Latency: ${payload.latency_ms} ms</span>
                <span class="badge"><i class="fa-solid fa-microchip"></i> Prompt Tokens: ${payload.prompt_tokens}</span>
                <span class="badge"><i class="fa-solid fa-chart-line"></i> Completion Tokens: ${payload.completion_tokens}</span>
            </div>
        </div>
    `;
    container.appendChild(wrapper);
}

async function uploadAdminDocument() {
    const targetFileInput = document.getElementById("documentFile");
    const targetKeyInput = document.getElementById("adminToken");
    
    // INSULATION FIX: Safely pull the specific index binary file out of the files array
    const selectedPdfFile = targetFileInput.files[0]; 
    const submittedAdminKey = targetKeyInput.value.trim();

    // Enforce immediate client-side validation guard rules
    if (!selectedPdfFile) {
        alert("⚠️ Please select a valid PDF lecture or manual file first.");
        return;
    }
    if (!submittedAdminKey) {
        alert("⚠️ Please enter your secure X-Admin-Key authorization token.");
        return;
    }

    const multipartFormData = new FormData();
    multipartFormData.append("file", selectedPdfFile);

    try {
        // TELEMETRY FIX: Clean browser console log replaces any python logger strings
        console.log("🚀 [UI Client] Submitting multipart stream to FastAPI backend gateway...");
        
        const apiResponse = await fetch("/api/ingest", {
            method: "POST",
            headers: { "X-Admin-Key": submittedAdminKey },
            body: multipartFormData
        });

        const jsonResponse = await apiResponse.json();
        
        if (apiResponse.ok) {
            alert(`✅ Success: ${jsonResponse.message}`);
            targetFileInput.value = ""; // Clear file selector on successful upload
        } else {
            alert(`❌ Ingestion Refused: ${jsonResponse.detail || "Authentication token validation failure."}`);
        }
    } catch (networkError) {
        console.error("🚨 Ingestion Request Crash:", networkError);
        alert(`❌ Network failure mapping sync parameters: ${networkError.message}`);
    }
}
