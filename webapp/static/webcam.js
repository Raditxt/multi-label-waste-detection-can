        // --- 1. Indikator Status Kamera ---
        const webcamImg = document.querySelector('.webcam-feed');
        const statusMessage = document.getElementById('statusMessage');

        webcamImg.onload = function() {
            statusMessage.textContent = "Kamera Aktif.";
            statusMessage.classList.remove('loading');
            statusMessage.classList.add('active');
        };

        webcamImg.onerror = function() {
            statusMessage.textContent = "Error: Gagal mengakses kamera. Mohon refresh halaman.";
            statusMessage.classList.remove('loading');
            statusMessage.classList.add('error');
        };

        // --- Helper untuk pewarnaan confidence (visualisasi prediksi) ---
        function getColorByConfidence(confidence) {
            if (confidence >= 0.8) return 'var(--color-success)'; // Hijau gelap
            if (confidence >= 0.6) return 'var(--color-warning)'; // Oranye/kuning
            if (confidence >= 0.3) return 'var(--color-info)'; // Biru muda
            return 'var(--color-danger)'; // Merah (low confidence)
        }

        // --- 4. Notifikasi Live Prediksi (Teks) - Menggunakan Polling API ---
        const currentPredictionText = document.getElementById('currentPredictionText');

        async function fetchLatestPrediction() {
            try {
                const response = await fetch('/get_latest_webcam_prediction');
                const data = await response.json();
                
                if (data.detected_labels && Object.keys(data.detected_labels).length > 0) {
                    currentPredictionText.innerHTML = ''; // Hapus konten sebelumnya
                    Object.entries(data.detected_labels).forEach(([label, prob]) => {
                        const span = document.createElement('span');
                        const formattedLabel = label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                        span.textContent = `${formattedLabel}: ${(prob * 100).toFixed(1)}% `;
                        span.style.color = getColorByConfidence(prob);
                        span.style.fontWeight = 'bold';
                        currentPredictionText.appendChild(span);
                    });
                } else if (data.error) {
                    currentPredictionText.textContent = `Error: ${data.error}`;
                    currentPredictionText.style.color = 'var(--color-danger)';
                } else {
                    currentPredictionText.textContent = "Tidak Ditemukan Sampah Spesifik";
                    currentPredictionText.style.color = 'var(--color-info)';
                }
            } catch (error) {
                console.error("Error fetching live prediction:", error);
                currentPredictionText.textContent = "Gagal memuat prediksi.";
                currentPredictionText.style.color = 'var(--color-danger)';
            }
        }

        let pollingInterval = setInterval(fetchLatestPrediction, 2000); 
        fetchLatestPrediction(); 

        window.addEventListener('beforeunload', () => {
            clearInterval(pollingInterval);
            console.log("Polling interval dihentikan.");
        });


        // --- 2. Tombol "Tangkap Gambar Sekarang" ---
        const captureButton = document.getElementById('captureButton');
        const captureSpinner = document.getElementById('captureSpinner');
        const captureResult = document.getElementById('captureResult');

        captureButton.addEventListener('click', async () => {
            captureResult.textContent = '';
            captureResult.className = 'capture-result-message';
            captureSpinner.style.display = 'block';
            captureButton.disabled = true;
            captureButton.textContent = 'Memproses...';

            const canvas = document.createElement('canvas');
            canvas.width = webcamImg.naturalWidth || webcamImg.videoWidth || 640; 
            canvas.height = webcamImg.naturalHeight || webcamImg.videoHeight || 480;

            const ctx = canvas.getContext('2d');
            if (webcamImg.readyState >= 2) { 
                ctx.drawImage(webcamImg, 0, 0, canvas.width, canvas.height);
                
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob, 'webcam_capture.jpg');

                    try {
                        const response = await fetch('/api/predict', { 
                            method: 'POST',
                            body: formData
                        });

                        const result = await response.json();

                        if (response.ok) {
                            let resultText = "Hasil Tangkapan: ";
                            if (result.detected_labels && Object.keys(result.detected_labels).length > 0) {
                                const labelsArray = Object.entries(result.detected_labels)
                                    .map(([label, prob]) => {
                                        const formattedLabel = label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                        return `<span style="color:${getColorByConfidence(prob)}; font-weight:bold;">${formattedLabel}: ${(prob * 100).toFixed(1)}%</span>`;
                                    });
                                resultText += labelsArray.join(', ');
                                captureResult.className = 'capture-result-message success';
                            } else {
                                resultText += "<span style='color:var(--color-info); font-weight:bold;'>Tidak Ditemukan Sampah Spesifik</span>";
                                captureResult.className = 'capture-result-message success';
                            }
                            captureResult.innerHTML = resultText;
                        } else {
                            captureResult.textContent = `Error: ${result.error || 'Terjadi kesalahan saat deteksi.'}`;
                            captureResult.className = 'capture-result-message error';
                        }
                    } catch (error) {
                        console.error('Error during capture and prediction:', error);
                        captureResult.textContent = 'Gagal melakukan deteksi dari tangkapan.';
                        captureResult.className = 'capture-result-message error';
                    } finally {
                        captureSpinner.style.display = 'none';
                        captureButton.disabled = false;
                        captureButton.textContent = 'Tangkap Gambar & Prediksi Sekarang';
                    }
                }, 'image/jpeg', 0.9);
            } else {
                captureResult.textContent = 'Webcam belum siap, coba lagi.';
                captureResult.className = 'capture-result-message error';
                captureSpinner.style.display = 'none';
                captureButton.disabled = false;
                captureButton.textContent = 'Tangkap Gambar & Prediksi Sekarang';
            }
        });

        // --- JavaScript untuk Mengganti Sumber Kamera (Diadaptasi dari permintaan Anda) ---
        document.getElementById('sourceForm').addEventListener('submit', async (e) => {
            e.preventDefault(); // Mencegah form melakukan reload halaman
            const form = e.target;
            const sourceInput = document.getElementById('sourceInput'); // Gunakan ID input
            const sourceValue = sourceInput.value.trim(); // Ambil nilai input

            const setSourceStatus = document.getElementById('setSourceStatus'); // Ambil elemen status
            setSourceStatus.textContent = 'Mengubah sumber...';
            setSourceStatus.style.color = '#fbc02d'; // Kuning

            // Logika untuk mengubah tipe data sumber (integer atau string)
            const sourceToSend = isNaN(sourceValue) || sourceValue === '' ? sourceValue : parseInt(sourceValue);
            
            try {
                const res = await fetch('/set_video_source', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: sourceToSend })
                });

                const data = await res.json();
                
                if (res.ok) {
                    setSourceStatus.textContent = data.message;
                    setSourceStatus.style.color = 'var(--color-success)'; // Hijau
                    alert(data.message); // Notifikasi pop-up sesuai permintaan Anda

                    // PENTING: Reload gambar stream agar video feed memulai ulang dengan sumber baru
                    // Menambahkan timestamp untuk menghindari caching browser
                    webcamImg.src = "/video_feed?t=" + new Date().getTime();

                } else {
                    setSourceStatus.textContent = `Error: ${data.message}`;
                    setSourceStatus.style.color = 'var(--color-danger)'; // Merah
                    alert(`Error: ${data.message}`); // Notifikasi pop-up error
                }
            } catch (error) {
                console.error("Error setting video source:", error);
                setSourceStatus.textContent = 'Gagal mengirim permintaan ubah sumber.';
                setSourceStatus.style.color = 'var(--color-danger)';
                alert('Gagal mengubah sumber video. Silakan cek konsol browser.');
            }
        });