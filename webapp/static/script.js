// webapp/static/script.js

// Fungsi untuk menangani submit form
document.getElementById('uploadForm').addEventListener('submit', function() {
    const button = document.getElementById('predictButton');
    const spinner = document.getElementById('loadingSpinner');
    
    // Tampilkan spinner
    spinner.style.display = 'inline-block';
    // Nonaktifkan tombol untuk mencegah double-submit
    button.disabled = true;
    button.classList.add('loading');
    
    // Ubah teks tombol menjadi "Memproses..."
    // Ini adalah cara yang lebih tangguh untuk mengubah teks tanpa menghapus spinner
    // Kita menyimpan teks asli dan menempatkan spinner setelahnya
    const originalButtonText = button.innerHTML; // Simpan innerHTML asli
    button.innerHTML = 'Memproses... '; // Ubah teks
    button.appendChild(spinner); // Pastikan spinner masih di dalam tombol
    spinner.style.display = 'inline-block'; // Pastikan spinner terlihat
});


// Fungsi untuk menyembunyikan flash messages secara otomatis
setTimeout(function() {
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(msg => {
        // Tambahkan kelas untuk memulai animasi fade out
        msg.style.opacity = '0';
        msg.style.transition = 'opacity 0.5s ease-out';
        // Hapus elemen dari DOM setelah transisi selesai
        setTimeout(() => msg.remove(), 500); 
    });
}, 5000); // Pesan akan hilang setelah 5 detik (5000 milidetik)