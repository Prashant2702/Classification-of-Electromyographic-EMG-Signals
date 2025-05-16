// Dark mode toggle functionality
const darkModeToggle = document.getElementById('darkModeToggle');
if (darkModeToggle) {
    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    });
}

// Check for saved dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}

// Real-time EMG Visualization
function initSignalVisualization() {
    const canvas = document.getElementById('signalCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const signalStrength = document.getElementById('signalStrength');
    const muscleGrid = document.getElementById('muscleGrid');
    let animationId = null;
    let isRunning = false;

    // Set canvas dimensions
    function resizeCanvas() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Muscle groups data
    const muscles = [
        { name: 'Biceps', active: false },
        { name: 'Triceps', active: false },
        { name: 'Deltoids', active: false },
        { name: 'Pectorals', active: false },
        { name: 'Quadriceps', active: false },
        { name: 'Hamstrings', active: false },
        { name: 'Abdominals', active: false },
        { name: 'Trapezius', active: false }
    ];

    // Create muscle cards
    function renderMuscleGrid() {
        muscleGrid.innerHTML = muscles.map(muscle => `
            <div class="muscle-card ${muscle.active ? 'active' : ''}">
                <h5>${muscle.name}</h5>
                <div class="activation-level">
                    <div class="activation-level-fill" style="width: ${muscle.active ? '70%' : '0%'}"></div>
                </div>
            </div>
        `).join('');
    }

    // Simulate EMG signal
    function simulateSignal(timestamp) {
        if (!isRunning) return;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.strokeStyle = '#6a11cb';
        ctx.lineWidth = 2;
        
        const points = [];
        const amplitude = 0.5 + Math.random() * 0.5;
        const frequency = 0.02;
        const noiseLevel = 0.3;
        
        for (let x = 0; x < canvas.width; x += 2) {
            const y = canvas.height / 2 + 
                      Math.sin(x * frequency + timestamp * 0.005) * amplitude * (canvas.height / 3) +
                      (Math.random() - 0.5) * noiseLevel * (canvas.height / 2);
            points.push({x, y});
        }

        // Draw signal
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();

        // Update signal strength
        const strength = Math.floor(amplitude * 100);
        signalStrength.style.width = `${strength}%`;
        
        // Randomly activate muscles
        if (timestamp % 1000 < 16) { // ~60fps
            muscles.forEach(muscle => {
                muscle.active = Math.random() > 0.7;
            });
            renderMuscleGrid();
        }

        animationId = requestAnimationFrame(simulateSignal);
    }

    // Event listeners
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            isRunning = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            renderMuscleGrid();
            simulateSignal(0);
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            isRunning = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            cancelAnimationFrame(animationId);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            signalStrength.style.width = '0%';
        });
    }
}

// Initialize visualizations when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    initSignalVisualization();
    
    // Classifier Page Functionality
    if (document.getElementById('analyzeBtn')) {
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            const fileInput = document.getElementById('emgUpload');
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                const formData = new FormData();
                formData.append('file', file);

                // Show loading state
                const resultsContainer = document.getElementById('resultsContainer');
                resultsContainer.innerHTML = '<div class="loading">Analyzing EMG data...</div>';

                fetch('http://localhost:5000/predict', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'Accept': 'application/json'
                    }
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        resultsContainer.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }

                    // Show/hide same dataset notice
                    const notice = document.getElementById('sameDatasetNotice');
                    if (data.is_same_dataset) {
                        notice.style.display = 'block';
                    } else {
                        notice.style.display = 'none';
                    }

                    // Display results
                    resultsContainer.innerHTML = `
                        <div class="result-card">
                            <h4>Analysis Results</h4>
                            <p>File: ${file.name}</p>
                            <div class="muscle-activation">
                                ${Object.entries(data.muscle_activation).map(([muscle, value]) => `
                                    <div class="muscle-row">
                                        <span class="muscle-name">${muscle}:</span>
                                        <div class="activation-bar">
                                            <div class="activation-fill" style="width: ${value * 100}%"></div>
                                        </div>
                                        <span class="activation-value">${(value * 100).toFixed(1)}%</span>
                                    </div>
                                `).join('')}
                            </div>
                            <canvas id="resultChart"></canvas>
                        </div>
                    `;

                    // Initialize chart if available
                    if (typeof Chart !== 'undefined') {
                        new Chart(document.getElementById('resultChart'), {
                            type: 'line',
                            data: {
                                labels: data.prediction.map((_, i) => i),
                                datasets: [{
                                    label: 'EMG Signal',
                                    data: data.prediction,
                                    borderColor: '#6a11cb',
                                    tension: 0.1
                                }]
                            }
                        });
                    }
                })
                .catch(error => {
                    resultsContainer.innerHTML = `<div class="error">Analysis failed: ${error.message}</div>`;
                });
            }
        });
    }

    // Visualization Page Functionality
    if (document.getElementById('updateVis')) {
        document.getElementById('updateVis').addEventListener('click', () => {
            const visType = document.getElementById('visType').value;
            const chart = document.getElementById('emgChart');
            
            if (typeof Chart !== 'undefined') {
                // Destroy existing chart if it exists
                if (chart.chart) {
                    chart.chart.destroy();
                }
                
                // Create new chart based on selected type
                chart.chart = new Chart(chart, {
                    type: visType === 'spectrogram' ? 'bar' : 'line',
                    data: {
                        labels: Array.from({length: 50}, (_, i) => i),
                        datasets: [{
                            label: 'EMG Data',
                            data: Array.from({length: 50}, () => Math.random() * 100),
                            borderColor: '#2575fc',
                            backgroundColor: visType === 'spectrogram' ? '#2575fc' : 'transparent'
                        }]
                    }
                });
            }
        });
    }

    // Form validation for login/signup pages
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', (e) => {
            const inputs = form.querySelectorAll('input[required]');
            let isValid = true;
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    input.style.borderColor = 'red';
                    isValid = false;
                } else {
                    input.style.borderColor = '';
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all required fields');
            }
        });
    });
});
