// EMG Visualization Module
class EMGVisualizer {
    constructor() {
        this.charts = {};
        this.data = null;
        this.initCharts();
        this.setupEventListeners();
    }

    initCharts() {
        // Initialize all chart canvases
        this.charts.timeChart = this.createChart('timeChart', 'line');
        this.charts.freqChart = this.createChart('freqChart', 'bar');
        this.charts.featureChart = this.createChart('featureChart', 'radar');
        this.charts.spectrogram = this.createChart('emgChart', 'line');
    }

    createChart(canvasId, type) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: type,
            data: { datasets: [] },
            options: this.getChartOptions(type)
        });
    }

    getChartOptions(type) {
        const baseOptions = {
            responsive: true,
            maintainAspectRatio: false
        };

        switch(type) {
            case 'line':
                return {
                    ...baseOptions,
                    scales: { y: { beginAtZero: false } }
                };
            case 'bar':
                return {
                    ...baseOptions,
                    scales: { y: { beginAtZero: true } }
                };
            case 'radar':
                return {
                    ...baseOptions,
                    scales: { r: { beginAtZero: true } }
                };
            default:
                return baseOptions;
        }
    }

    setupEventListeners() {
        document.getElementById('updateVis').addEventListener('click', () => this.updateVisualization());
        document.getElementById('loadData').addEventListener('click', () => this.loadEMGData());
    }

    async loadEMGData() {
        try {
            const response = await fetch('EMG_Numerical_Data.csv');
            const csvData = await response.text();
            this.processData(csvData);
            this.updateAllCharts();
        } catch (error) {
            console.error('Error loading EMG data:', error);
        }
    }

    processData(csvData) {
        // Simple CSV parsing (would need enhancement for production)
        const lines = csvData.split('\n');
        const headers = lines[0].split(',');
        const values = lines.slice(1).map(line => line.split(',').map(Number));
        
        this.data = {
            headers,
            values,
            timeSeries: values.map(row => row.slice(0, -1)), // All columns except last
            labels: values.map(row => row[row.length - 1])   // Last column
        };
    }

    updateAllCharts() {
        if (!this.data) return;

        // Update time domain chart
        this.updateTimeChart();
        
        // Update frequency domain chart
        this.updateFrequencyChart();
        
        // Update feature chart
        this.updateFeatureChart();
    }

    updateTimeChart() {
        const datasets = this.data.headers.slice(0, -1).map((header, i) => ({
            label: header,
            data: this.data.timeSeries.map(row => row[i]),
            borderColor: this.getChannelColor(i),
            tension: 0.1
        }));

        this.charts.timeChart.data.labels = Array.from(
            {length: this.data.timeSeries.length}, 
            (_, i) => i
        );
        this.charts.timeChart.data.datasets = datasets;
        this.charts.timeChart.update();
    }

    updateFrequencyChart() {
        // Simple FFT simulation (would use real FFT in production)
        const fftData = this.data.timeSeries.map(channel => 
            channel.map((val, i) => Math.abs(val * Math.sin(i * 0.1)))
        );

        const datasets = this.data.headers.slice(0, -1).map((header, i) => ({
            label: header,
            data: fftData[i],
            backgroundColor: this.getChannelColor(i),
            borderColor: this.getChannelColor(i),
            borderWidth: 1
        }));

        this.charts.freqChart.data.labels = Array.from(
            {length: fftData[0].length}, 
            (_, i) => `${(i * 10)}Hz`
        );
        this.charts.freqChart.data.datasets = datasets;
        this.charts.freqChart.update();
    }

    updateFeatureChart() {
        // Calculate simple features (would use proper feature extraction in production)
        const features = this.data.headers.slice(0, -1).map((header, i) => {
            const channel = this.data.timeSeries.map(row => row[i]);
            const mean = channel.reduce((a, b) => a + b, 0) / channel.length;
            const rms = Math.sqrt(channel.reduce((a, b) => a + b * b, 0) / channel.length);
            return { mean, rms };
        });

        this.charts.featureChart.data = {
            labels: ['Mean', 'RMS'],
            datasets: this.data.headers.slice(0, -1).map((header, i) => ({
                label: header,
                data: [features[i].mean, features[i].rms],
                backgroundColor: this.getChannelColor(i, 0.2),
                borderColor: this.getChannelColor(i),
                borderWidth: 1
            }))
        };
        this.charts.featureChart.update();
    }

    getChannelColor(index, opacity = 1) {
        const colors = [
            `rgba(75, 192, 192, ${opacity})`,
            `rgba(54, 162, 235, ${opacity})`,
            `rgba(255, 99, 132, ${opacity})`,
            `rgba(255, 159, 64, ${opacity})`,
            `rgba(153, 102, 255, ${opacity})`,
            `rgba(201, 203, 207, ${opacity})`,
            `rgba(255, 205, 86, ${opacity})`,
            `rgba(75, 192, 192, ${opacity})`
        ];
        return colors[index % colors.length];
    }

    updateVisualization() {
        const visType = document.getElementById('visType').value;
        // Show/hide charts based on selection
        document.querySelectorAll('.chart-container').forEach(container => {
            container.style.display = 'none';
        });

        if (visType === 'time') {
            document.getElementById('timeChart').parentElement.style.display = 'block';
        } else if (visType === 'frequency') {
            document.getElementById('freqChart').parentElement.style.display = 'block';
        } else if (visType === 'features') {
            document.getElementById('featureChart').parentElement.style.display = 'block';
        } else if (visType === 'spectrogram') {
            document.getElementById('emgChart').parentElement.style.display = 'block';
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EMGVisualizer();
});
