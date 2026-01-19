// Example RPM Line Chart
new Chart(document.getElementById("rpmChart"), {
    type: "line",
    data: {
        labels: ["1", "2", "3", "4", "5"],
        datasets: [{
            label: "Rotor RPM",
            data: [20, 30, 40, 35, 50],
            borderColor: "cyan",
            borderWidth: 2
        }]
    }
});

// Temperature Gauge
new Chart(document.getElementById("tempChart"), {
    type: "bar",
    data: {
        labels: ["Gearbox Temp"],
        datasets: [{
            label: "°C",
            data: [60],
            backgroundColor: "orange"
        }]
    }
});

// Vibration Trend
new Chart(document.getElementById("vibrationChart"), {
    type: "line",
    data: {
        labels: ["1", "2", "3", "4"],
        datasets: [{
            label: "Vibration Level",
            data: [1, 3, 5, 4],
            borderColor: "yellow"
        }]
    }
});

// Risk Meter
new Chart(document.getElementById("riskChart"), {
    type: "doughnut",
    data: {
        labels: ["Risk", "Safe"],
        datasets: [{
            data: [30, 70],
            backgroundColor: ["red", "green"]
        }]
    }
});
