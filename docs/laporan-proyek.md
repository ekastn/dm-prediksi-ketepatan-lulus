# Laporan Proyek Data Mining

**Prediksi Ketepatan Lulus Mahasiswa Menggunakan Decision Tree Classifier dengan Pendekatan CRISP-DM**

---

# Bab I - Pendahuluan

## 1.1 Latar Belakang

Ketepatan waktu kelulusan mahasiswa merupakan salah satu indikator utama dalam penilaian akreditasi perguruan tinggi. Badan Akreditasi Nasional Perguruan Tinggi (BAN-PT) menjadikan persentase kelulusan tepat waktu sebagai salah satu kriteria evaluasi mutu program studi. Semakin tinggi proporsi mahasiswa yang menyelesaikan studi dalam batas waktu yang ditetapkan, semakin baik gambaran kualitas tata kelola akademik suatu institusi.

Sistem informasi akademik pada institusi perguruan tinggi saat ini telah mampu mendokumentasikan berbagai aktivitas dan rekam jejak nilai mahasiswa secara digital. Setiap semester, data berupa Indeks Prestasi Semester (IPS), Indeks Prestasi Kumulatif (IPK), serta jumlah Satuan Kredit Semester (SKS) yang diambil oleh mahasiswa terus bertambah dan tersimpan di dalam basis data pusat. Namun, aset data akademik yang sangat besar ini umumnya hanya diperlakukan sebagai arsip digital pasif. Data tersebut sekadar digunakan untuk keperluan pelaporan administratif atau pencetakan Kartu Hasil Studi (KHS) dan Transkrip Nilai. Potensi tersembunyi dari data historis kelulusan mahasiswa angkatan-angkatan sebelumnya belum dimanfaatkan secara optimal untuk mendukung pengambilan keputusan strategis program studi.

Salah satu pendekatan yang dapat digunakan untuk memanfaatkan data historis tersebut adalah machine learning, khususnya algoritma klasifikasi Decision Tree. Decision Tree dipilih karena memiliki keunggulan dalam hal interpretabilitas, yaitu model yang dihasilkan berupa pohon keputusan dan aturan logika (decision rules) yang mudah dipahami oleh pihak non-teknis seperti manajemen kampus dan bagian akademik. Dengan demikian, institusi tidak hanya menerima hasil prediksi, tetapi juga memahami faktor-faktor akademik apa yang menyebabkan seorang mahasiswa berisiko tidak lulus tepat waktu.

Penelitian ini menggunakan data akademik mahasiswa program vokasi (D3) dan sarjana (S1) yang berasal dari database akademik Universitas Logistik dan Bisnis Internasional. Data yang digunakan dibatasi pada performa akademik tiga semester pertama untuk menghindari kebocoran data (data leakage) yang dapat menyebabkan hasil prediksi menjadi bias. Dengan pendekatan CRISP-DM (Cross-Industry Standard Process for Data Mining), penelitian ini bertujuan untuk membangun model prediksi yang dapat mengklasifikasikan mahasiswa ke dalam kategori "Tepat Waktu" atau "Terlambat" serta menghasilkan aturan keputusan yang dapat digunakan sebagai sistem peringatan dini (early-warning system) oleh pihak program studi.

## 1.2 Identifikasi Masalah

Berdasarkan latar belakang yang telah dipaparkan, terdapat lima masalah utama yang diidentifikasi dalam penelitian ini:

**1. Ketidakseimbangan kelas pada data kelulusan mahasiswa**

Dataset yang digunakan memiliki distribusi kelas yang timpang. Dari total 608 mahasiswa, sebanyak 540 mahasiswa (88,8%) dinyatakan lulus tepat waktu, sementara hanya 68 mahasiswa (11,2%) yang lulus terlambat. Ketimpangan ini menjadi tantangan serius bagi model klasifikasi, terutama algoritma Decision Tree yang cenderung bias terhadap kelas mayoritas. Tanpa penanganan yang tepat, model akan kesulitan mempelajari pola kelas minoritas yang justru menjadi fokus utama prediksi, yaitu mahasiswa yang berisiko tidak lulus tepat waktu.

**2. Kualitas data akademik yang belum optimal**

Database akademik yang digunakan merupakan hasil migrasi dari sistem lama ke sistem baru. Proses migrasi ini meninggalkan sejumlah masalah kualitas data yang signifikan. Pertama, nilai IPS sebesar 0.0 ditemukan pada 219 mahasiswa (36% dari total) yang bukan merupakan nilai aktual melainkan placeholder dari sistem legacy. Kedua, bug pada kolom TSKS (Total SKS Semester) menyebabkan nilai SKS untuk angkatan 2020+ tersimpan secara kumulatif bukan per-semester, dengan nilai mencapai 80-133 untuk semester 1-2 yang seharusnya bernilai wajar 4-24. Ketiga, terdapat missing values pada data IPS di angkatan-angkatan awal yang memerlukan strategi imputasi yang tepat.

**3. Belum diketahui fitur akademik yang paling berpengaruh terhadap ketepatan lulus**

Meskipun sistem akademik menyimpan beragam data akademik seperti IPS, SKS, dan nilai mata kuliah, belum diketahui secara pasti fitur-fitur mana yang memiliki kontribusi terbesar dalam memprediksi ketepatan lulus. Identifikasi fitur dominan diperlukan agar model yang dihasilkan tidak hanya akurat tetapi juga memberikan wawasan yang dapat ditindaklanjuti oleh pihak akademik, misalnya dalam bentuk aturan keputusan yang jelas.

**4. Belum diketahui kemampuan generalisasi Decision Tree terhadap mahasiswa angkatan baru**

Model prediksi yang dilatih pada data historis belum tentu dapat menggeneralisasi dengan baik pada mahasiswa angkatan baru yang memiliki karakteristik berbeda. Pertanyaan mengenai temporal validity apakah model yang sama tetap berlaku ketika dihadapkan pada cohort mahasiswa di luar periode pelatihan masih belum terjawab. Hal ini penting karena dalam skenario deployment sesungguhnya, model akan digunakan untuk memprediksi mahasiswa angkatan baru yang pola akademiknya mungkin berbeda dari data historis.

**5. Belum diketahui konfigurasi hyperparameter optimal untuk Decision Tree pada dataset ini**

Decision Tree memiliki sejumlah hyperparameter seperti max_depth, min_samples_leaf, dan criterion yang memengaruhi kompleksitas dan performa model. Konfigurasi hyperparameter yang optimal untuk dataset akademik dengan ketidakseimbangan kelas tinggi ini belum diketahui dan memerlukan pencarian secara sistematis melalui hyperparameter tuning.

## 1.3 Tujuan Penelitian

Berdasarkan identifikasi masalah yang telah diuraikan, tujuan yang ingin dicapai dalam penelitian ini adalah sebagai berikut:

1. Membangun model klasifikasi Decision Tree untuk memprediksi ketepatan lulus mahasiswa berdasarkan performa akademik tiga semester pertama dengan menggunakan pendekatan CRISP-DM.

2. Mengidentifikasi fitur-fitur akademik yang paling berpengaruh terhadap ketepatan lulus melalui analisis feature importance dan decision rules yang dihasilkan oleh algoritma Decision Tree.

3. Melakukan hyperparameter tuning secara sistematis menggunakan GridSearchCV dengan k-fold cross-validation untuk mendapatkan konfigurasi hyperparameter Decision Tree yang optimal pada dataset dengan ketidakseimbangan kelas.

4. Mengevaluasi kemampuan generalisasi model menggunakan dua pendekatan validasi, yaitu stratified split (pembagian acak dengan mempertahankan proporsi kelas) dan temporal split (pelatihan pada data historis dan pengujian pada angkatan baru) untuk mengukur kesenjangan performa antara kondisi eksperimental dan skenario deployment nyata.

5. Menghasilkan aturan keputusan (decision rules) yang interpretable dari model Decision Tree yang dapat digunakan oleh bagian akademik sebagai sistem peringatan dini (early-warning system) untuk mendeteksi mahasiswa yang berisiko tidak lulus tepat waktu.

## 1.4 Manfaat Penelitian

Penelitian ini diharapkan dapat memberikan manfaat bagi empat pihak utama, yaitu:

**Bagi institusi perguruan tinggi**, penelitian ini menyediakan sistem peringatan dini (early-warning system) berbasis data yang memungkinkan intervensi akademik sebelum mahasiswa terlambat lulus. Institusi dapat secara proaktif mengidentifikasi mahasiswa berisiko tinggi dan memberikan pendampingan akademik, konseling, atau penyesuaian beban studi sejak dini.

**Bagi program studi**, hasil penelitian memberikan informasi berbasis data untuk evaluasi kurikulum dan kebijakan beban SKS. Aturan keputusan yang dihasilkan (seperti pola "overload lalu collapse" pada SKS semester 2 dan 3) dapat menjadi masukan objektif untuk meninjau kembali kebijakan pengambilan SKS maksimum di semester awal.

**Bagi mahasiswa**, penelitian ini memberikan gambaran risiko ketidaktepatan lulus berdasarkan pola akademik tiga semester pertama. Dengan mengetahui faktor-faktor yang berkontribusi terhadap risiko keterlambatan, mahasiswa dapat lebih waspada dan merencanakan strategi studi yang lebih efektif sejak awal perkuliahan.

**Bagi akademisi dan peneliti lain**, penelitian ini menjadi referensi penerapan algoritma Decision Tree pada data akademik dengan ketidakseimbangan kelas tinggi dan tantangan validasi temporal. Temuan mengenai kesenjangan performa antara stratified split dan temporal split, serta distribusi SKS antar angkatan, memberikan pelajaran berharga bagi penelitian serupa di masa mendatang.

## 1.5 Batasan Masalah

Agar penelitian tetap terfokus dan tidak meluas, ditetapkan batasan-batasan sebagai berikut:

1. Data yang digunakan terbatas pada mahasiswa program vokasi (D3) dan sarjana (S1) di lingkungan Universitas Logistik dan Bisnis Internasional, dengan rentang angkatan 2015 hingga 2023. Setelah melalui proses filtering dan preprocessing, dataset final terdiri dari 608 mahasiswa.

2. Fitur prediktor yang digunakan hanya berasal dari data akademik tiga semester pertama, yaitu Indeks Prestasi Semester (IPS), jumlah Satuan Kredit Semester (SKS), mata kuliah yang diulang (failed courses), serta fitur turunannya. Data semester 4 dan seterusnya tidak digunakan karena bersifat circular terhadap target (leakage).

3. Algoritma klasifikasi yang digunakan terbatas pada Decision Tree Classifier dari pustaka scikit-learn. Penelitian ini tidak membandingkan Decision Tree dengan algoritma lain seperti Random Forest, SVM, atau Neural Network.

4. Metodologi yang digunakan adalah CRISP-DM (Cross-Industry Standard Process for Data Mining), mulai dari business understanding hingga evaluation. Fase deployment tidak termasuk dalam lingkup penelitian ini.

5. Evaluasi model dilakukan pada dua skenario validasi: stratified split dengan proporsi 80/20 yang mempertahankan distribusi kelas, dan temporal split dengan data latih angkatan <=2021 (377 mahasiswa) dan data uji angkatan >2021 (231 mahasiswa). Validasi tambahan menggunakan Repeated Stratified K-Fold 10 kali ulang dengan 10 fold untuk estimasi performa yang lebih robust.

6. Penelitian ini berfokus pada aspek prediktif dan interpretabilitas model. Implementasi sistem informasi, integrasi dengan database akademik, dan pengembangan antarmuka pengguna tidak termasuk dalam lingkup penelitian.

---

# Bab II - Tinjauan Pustaka dan Landasan Teori

## 2.1 Knowledge Discovery in Databases (KDD) dan CRISP-DM

Data mining adalah proses ekstraksi pola, hubungan, atau pengetahuan yang sebelumnya tidak diketahui, valid, dan berpotensi berguna dari sekumpulan data yang besar [X]. Istilah ini merujuk pada langkah inti dalam keseluruhan proses Knowledge Discovery in Databases (KDD) yang diperkenalkan oleh Fayyad et al. [X]. Data mining menggabungkan teknik dari statistik, machine learning, dan sistem basis data untuk menemukan struktur tersembunyi dalam data yang dapat digunakan untuk prediksi, klasifikasi, clustering, atau asosiasi.

Dalam konteks penelitian ini, data mining diterapkan untuk mengekstraksi pola akademik dari data historis mahasiswa yang tersimpan di database perguruan tinggi. Pola yang ditemukan berupa aturan keputusan yang menghubungkan performa akademik tiga semester pertama dengan status ketepatan lulus. Pola ini tidak dapat diperoleh melalui query SQL sederhana atau laporan statistik konvensional karena melibatkan interaksi non-linear antar fitur seperti SKS, IPS, dan mata kuliah gagal.

Cross-Industry Standard Process for Data Mining (CRISP-DM) adalah framework proses data mining yang dikembangkan oleh konsorsium perusahaan pada tahun 1999 yang melibatkan DaimlerChrysler, SPSS, dan NCR [X]. CRISP-DM menjadi standar de facto dalam industri data mining karena pendekatannya yang terstruktur, iteratif, dan independen terhadap industri atau teknologi tertentu.

CRISP-DM terdiri dari enam fase utama yang membentuk siklus hidup proyek data mining [X]:

1. **Business Understanding (Pemahaman Bisnis).** Fase ini berfokus pada pemahaman tujuan proyek dari perspektif bisnis, kemudian mentransformasikannya menjadi masalah data mining. Dalam penelitian ini, fase menghasilkan identifikasi bahwa institusi membutuhkan sistem peringatan dini untuk mendeteksi mahasiswa berisiko tidak lulus tepat waktu.

2. **Data Understanding (Pemahaman Data).** Fase ini melibatkan pengumpulan data awal, deskripsi data, eksplorasi, dan verifikasi kualitas data. Pada proyek ini, fase mencakup koneksi ke database SQL Server LITIGASI, profiling 405 tabel, identifikasi 4 tabel relevan (`tblMHS`, `IPSIPK`, `Qnilai_mhs`, `Kul_Kehadiran`), serta penemuan masalah kualitas data seperti placeholder IPS=0.0, bug SKS kumulatif, dan ketidakcocokan format NIM.

3. **Data Preparation (Persiapan Data).** Fase ini mencakup semua aktivitas untuk membangun dataset final dari data mentah. Meliputi pembersihan data (replacement IPS=0.0 menjadi NaN), imputasi missing values, pembuatan fitur turunan (derived features), filtering mahasiswa, dan pembagian dataset menjadi data latih dan data uji. Dataset final berisi 608 mahasiswa dengan 14 fitur prediktor.

4. **Modeling (Pemodelan).** Fase ini memilih dan menerapkan teknik pemodelan. Penelitian ini menggunakan Decision Tree Classifier dari scikit-learn dengan tiga iterasi modeling: baseline Decision Tree, global median imputation, dan hyperparameter tuning menggunakan GridSearchCV.

5. **Evaluation (Evaluasi).** Fase ini mengevaluasi model untuk memastikan bahwa model telah mencapai tujuan bisnis. Evaluasi dilakukan melalui temporal validation, repeated stratified k-fold cross-validation, error analysis, dan permutation importance.

6. **Deployment (Penyebaran).** Fase ini menerapkan model ke dalam lingkungan produksi. Fase ini tidak termasuk dalam lingkup penelitian ini karena model masih memerlukan perbaikan signifikan sebelum deployment, sebagaimana ditunjukkan oleh hasil evaluasi temporal.

Selain CRISP-DM, terdapat dua framework data mining lain yang umum digunakan:

**SEMMA (Sample, Explore, Modify, Model, Assess)** adalah metodologi yang dikembangkan oleh SAS Institute [X]. SEMMA lebih berfokus pada aspek teknis pemodelan dan tidak mencakup fase business understanding dan deployment secara eksplisit. SEMMA cocok untuk proyek data mining yang tujuan bisnisnya sudah jelas dan hanya membutuhkan panduan teknis.

**KDD (Knowledge Discovery in Databases)** adalah proses yang lebih luas yang mencakup sembilan langkah: memahami domain, membuat target dataset, preprocessing, transformasi, data mining, interpretasi, dan penerapan pengetahuan [X]. KDD lebih akademik dan komprehensif, namun kurang memberikan panduan praktis untuk proyek industri [X].

CRISP-DM dipilih dalam penelitian ini karena tiga alasan utama. Pertama, CRISP-DM mencakup fase business understanding dan evaluation secara eksplisit, yang sangat relevan untuk penelitian yang bertujuan menghasilkan aturan keputusan yang dapat digunakan oleh pihak non-teknis. Kedua, CRISP-DM bersifat iteratif, memungkinkan perbaikan antar fase seperti yang terjadi dalam penelitian ini ketika ditemukan proxy temporal pada imputasi per-angkatan yang memerlukan kembali ke fase data preparation. Ketiga, CRISP-DM adalah standar industri yang paling banyak diadopsi, sehingga hasil penelitian dapat dengan mudah direproduksi dan dibandingkan dengan studi serupa.

## 2.2 Klasifikasi dalam Machine Learning

Supervised learning adalah paradigma machine learning di mana model dilatih menggunakan data berlabel, yaitu setiap instance data memiliki pasangan fitur (input) dan label (target) yang diketahui [X]. Tujuan dari supervised learning adalah mempelajari fungsi pemetaan dari fitur ke target sehingga model dapat memprediksi label dari data baru yang belum pernah dilihat sebelumnya.

Supervised learning terbagi menjadi dua kategori utama berdasarkan tipe targetnya: klasifikasi dan regresi. Klasifikasi digunakan ketika target berupa variabel diskrit atau kategorikal, sedangkan regresi digunakan ketika target berupa variabel kontinu [X]. Contoh klasifikasi adalah memprediksi apakah seorang mahasiswa lulus tepat waktu (Tepat/Tidak Tepat), sedangkan contoh regresi adalah memprediksi Indeks Prestasi Kumulatif (IPK) dalam rentang 0.0 sampai 4.0. Penelitian ini termasuk dalam kategori klasifikasi karena target berupa label biner: tepat waktu (1) atau tidak tepat waktu (0).

Binary classification adalah jenis klasifikasi dengan dua kelas target [X]. Dalam penelitian ini, kelas target adalah "Tepat Waktu" (mayoritas) dan "Tidak Tepat Waktu" (minoritas). Model dilatih untuk memisahkan kedua kelas berdasarkan pola pada 14 fitur akademik.

Tantangan utama dalam binary classification pada dataset ini adalah class imbalance, yaitu kondisi di mana jumlah sampel antar kelas tidak seimbang [X]. Dataset memiliki 540 mahasiswa tepat waktu (88.8%) dan hanya 68 mahasiswa tidak tepat waktu (11.2%). Ketidakseimbangan ini menyebabkan model cenderung bias ke kelas mayoritas karena fungsi loss standar memberikan bobot yang sama pada setiap sampel. Akibatnya, model dapat mencapai akurasi tinggi (88.8%) hanya dengan memprediksi semua mahasiswa sebagai "Tepat Waktu", tanpa pernah mendeteksi mahasiswa berisiko.

Dalam konteks prediksi ketepatan lulus, kelas minoritas justru memiliki nilai bisnis yang lebih tinggi. Kesalahan mendeteksi mahasiswa berisiko (false negative) lebih mahal daripada kesalahan menandai mahasiswa aman sebagai berisiko (false positive). Oleh karena itu, pemilihan metrik evaluasi yang tepat menjadi krusial.

Confusion matrix adalah tabel yang menggambarkan performa model klasifikasi dengan membandingkan prediksi model dengan nilai aktual [X]. Untuk binary classification, confusion matrix berbentuk matriks 2x2:

| | Prediksi Negatif (0) | Prediksi Positif (1) |
|---|---|---|
| **Aktual Negatif (0)** | True Negative (TN) | False Positive (FP) |
| **Aktual Positif (1)** | False Negative (FN) | True Positive (TP) |

Dalam konteks penelitian ini:
- **True Negative (TN):** Mahasiswa tidak tepat waktu yang diprediksi tidak tepat waktu (deteksi benar).
- **True Positive (TP):** Mahasiswa tepat waktu yang diprediksi tepat waktu.
- **False Negative (FN):** Mahasiswa tidak tepat waktu yang diprediksi tepat waktu (terlewatkan).
- **False Positive (FP):** Mahasiswa tepat waktu yang diprediksi tidak tepat waktu (salah alarm).

Dari confusion matrix, diturunkan berbagai metrik evaluasi [X]:

**Accuracy** mengukur proporsi prediksi benar dari total prediksi:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

Accuracy merupakan metrik yang menyesatkan pada dataset tidak seimbang. Model yang memprediksi semua mahasiswa sebagai "Tepat Waktu" akan mencapai akurasi 88.8% tanpa mendeteksi satu pun mahasiswa berisiko.

**Precision (Positive Predictive Value)** mengukur proporsi prediksi positif yang benar:

$$Precision(1) = \frac{TP}{TP + FP}$$

Untuk kelas minoritas (0), precision dihitung dengan:

$$Precision(0) = \frac{TN}{TN + FN}$$

Precision(0) menjawab pertanyaan: dari semua mahasiswa yang diprediksi berisiko, berapa persen yang benar-benar berisiko?

**Recall (Sensitivity, True Positive Rate)** mengukur proporsi aktual positif yang berhasil dideteksi:

$$Recall(1) = \frac{TP}{TP + FN}$$

Untuk kelas minoritas (0):

$$Recall(0) = \frac{TN}{TN + FP}$$

Recall(0) menjawab pertanyaan: dari semua mahasiswa yang benar-benar berisiko, berapa persen yang berhasil dideteksi model?

**F1-Score** adalah harmonic mean dari precision dan recall:

$$F1(1) = 2 \times \frac{Precision(1) \times Recall(1)}{Precision(1) + Recall(1)}$$

Untuk kelas minoritas (0):

$$F1(0) = 2 \times \frac{Precision(0) \times Recall(0)}{Precision(0) + Recall(0)}$$

F1-Score memberikan keseimbangan antara precision dan recall, sehingga lebih informatif daripada accuracy pada dataset tidak seimbang.

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)** mengukur kemampuan model dalam membedakan kelas pada berbagai threshold [X]. AUC bernilai 0.5 untuk model acak dan 1.0 untuk model sempurna. AUC lebih robust terhadap class imbalance dibandingkan accuracy, namun tidak memberikan informasi langsung tentang performa pada threshold tertentu.

Dalam penelitian ini, metrik utama yang digunakan adalah Recall(0) dan F1(0), yaitu recall dan F1-Score untuk kelas minoritas (mahasiswa tidak tepat waktu). Pemilihan ini didasarkan pada pertimbangan bisnis dan teknis:

1. **Nilai bisnis kelas minoritas:** Mendeteksi mahasiswa berisiko adalah tujuan utama sistem early-warning. Melewatkan mahasiswa berisiko (false negative) berarti institusi kehilangan kesempatan untuk melakukan intervensi. Satu mahasiswa yang terdeteksi dapat dicegah dari keterlambatan lulus.

2. **Ketidakcukupan accuracy:** Dengan ketidakseimbangan 88.8% vs 11.2%, model yang memprediksi semua mahasiswa sebagai tepat waktu mencapai accuracy 88.8% namun recall(0)=0.000. Accuracy memberikan ilusi performa yang baik.

3. **Trade-off precision dan recall:** Precision(0) tinggi (sedikit false positive) lebih mudah dicapai dengan threshold konservatif, namun recall(0) rendah berarti banyak mahasiswa berisiko terlewat. Di sisi lain, recall(0) tinggi (sedikit false negative) cenderung menurunkan precision(0). F1(0) menyeimbangkan keduanya.

4. **Relasi dengan tujuan penelitian:** Tujuan penelitian adalah menghasilkan sistem early-warning. Sistem yang tidak pernah memberi peringatan (recall rendah) tidak berguna. Oleh karena itu, recall(0) dan F1(0) menjadi metrik yang lebih relevan daripada accuracy atau AUC.

Metrik Accuracy, Precision(1), dan AUC tetap dilaporkan sebagai referensi tambahan, namun keputusan pemilihan model terbaik didasarkan pada F1(0).

## 2.3 Decision Tree Classifier

Decision Tree adalah algoritma supervised learning yang bekerja dengan membagi data secara rekursif (recursive partitioning) berdasarkan nilai fitur hingga mencapai daerah keputusan yang homogen [X]. Setiap simpul internal pada pohon merepresentasikan pengujian terhadap suatu fitur, setiap cabang merepresentasikan hasil pengujian, dan setiap daun (leaf) merepresentasikan kelas keputusan.

Proses pembagian data dilakukan dengan memilih fitur dan threshold yang memaksimalkan pemisahan kelas. Dua kriteria pemisahan yang umum digunakan adalah Information Gain yang dihitung dari entropi, dan Gini impurity [X]. Algoritma CART (Classification and Regression Tree) yang diimplementasikan di scikit-learn menggunakan Gini impurity sebagai default:

$$Gini(t) = 1 - \sum_{j=1}^{k} p(j|t)^2$$

dimana $p(j|t)$ adalah proporsi sampel kelas $j$ pada simpul $t$. Nilai Gini berkisar antara 0 (semua sampel satu kelas, murni) hingga 0.5 (distribusi seragam). Algoritma memilih split yang meminimalkan rata-rata tertimbang Gini impurity dari dua simpul anak.

Dalam konteks penelitian ini, Decision Tree digunakan untuk mengklasifikasikan mahasiswa ke dalam "Tepat Waktu" atau "Tidak Tepat" berdasarkan 14 fitur akademik. Pohon yang dihasilkan memiliki kedalaman maksimum 3 dengan 7 daun, menghasilkan aturan keputusan seperti "jika SKS semester 2 lebih dari 18.5 dan SKS semester 3 kurang dari atau sama dengan 18.5, maka mahasiswa berisiko tidak lulus tepat waktu."

Decision Tree memiliki beberapa kelebihan yang relevan dengan penelitian ini [X]. Pertama, **interpretabilitas**: struktur pohon dan decision rules dapat dipahami oleh pihak non-teknis seperti bagian akademik. Kedua, **tidak memerlukan scaling**: Decision Tree tidak sensitif terhadap skala fitur, sehingga fitur IPS (0.0-4.0) dan SKS (1-24) dapat digunakan langsung tanpa normalisasi. Ketiga, **mampu menangani hubungan non-linear** antar fitur, seperti interaksi antara SKS semester 2, SKS semester 3, dan IPS dalam memprediksi kelulusan.

Di sisi lain, Decision Tree memiliki keterbatasan yang perlu diantisipasi [X]. Pertama, **rentan overfitting**: pohon yang terlalu dalam cenderung menghafal data latih, sebagaimana terjadi pada baseline dengan depth=8 dan train accuracy=1.0. Kedua, **sensitif terhadap data imbalance**: kecenderungan algoritma memilih split yang menguntungkan kelas mayoritas, menyebabkan bias terhadap kelas minoritas. Ketiga, **varians tinggi**: perubahan kecil pada data latih dapat mengubah struktur pohon secara signifikan, terutama pada dataset dengan sampel minoritas terbatas.

Hyperparameter utama Decision Tree meliputi `max_depth` yang membatasi kedalaman pohon untuk mencegah overfitting, `min_samples_leaf` yang menentukan jumlah minimum sampel pada setiap daun, `criterion` yang memilih fungsi pemisahan (Gini atau entropi), dan `min_impurity_decrease` yang menghentikan split jika penurunan impurity di bawah threshold [X]. Dalam penelitian ini, hyperparameter tersebut dioptimalkan melalui GridSearchCV untuk menemukan konfigurasi terbaik.

## 2.4 Hyperparameter Tuning dengan GridSearchCV

Dalam machine learning, terdapat perbedaan mendasar antara parameter model dan hyperparameter. **Parameter model** adalah nilai yang dipelajari secara otomatis dari data selama proses pelatihan, seperti threshold split pada Decision Tree. **Hyperparameter** adalah konfigurasi yang ditetapkan sebelum pelatihan dan mengontrol perilaku algoritma, seperti `max_depth` dan `min_samples_leaf` [X].

**GridSearchCV** adalah metode exhaustive search yang menguji seluruh kombinasi hyperparameter dalam grid yang telah ditentukan [X]. Setiap kombinasi dievaluasi menggunakan cross-validation, dan kombinasi dengan performa terbaik dipilih sebagai konfigurasi optimal. GridSearchCV menjamin bahwa kombinasi terbaik dalam grid ditemukan, namun biaya komputasi meningkat secara eksponensial seiring bertambahnya jumlah hyperparameter.

**K-Fold Cross-Validation** membagi data latih menjadi k bagian (fold) secara berurutan. Model dilatih pada k-1 fold dan diuji pada 1 fold sisanya, diulang sebanyak k kali sehingga setiap fold menjadi data uji tepat satu kali. **Repeated Stratified K-Fold** mengulang proses k-fold sebanyak n kali dengan pengacakan berbeda, menghasilkan estimasi performa yang lebih stabil dan confidence interval yang lebih reliable [X]. Dalam penelitian ini, Repeated Stratified K-Fold digunakan dengan konfigurasi 10 kali ulang dan 10 fold (100 evaluasi total).

Strategi tuning yang diterapkan dalam penelitian ini mencakup tiga pendekatan. **Pre-pruning tuning** menggunakan GridSearchCV dengan 240 kombinasi hyperparameter (`max_depth`, `min_samples_leaf`, `min_samples_split`, `class_weight`, `criterion`) dan 5-fold cross-validation, menghasilkan 1.200 fit. Konfigurasi terbaik adalah `max_depth=3`, `min_samples_leaf=10`, dan `criterion=gini`. **Post-pruning** menggunakan `ccp_alpha` untuk menyederhanakan tree lebih lanjut setelah pre-pruning. **Combined tuning** menambahkan `max_features` dan `min_impurity_decrease` dengan 28 kombinasi tambahan, namun performa test tetap stagnan pada F1(0)=0.889, mengindikasikan bahwa ceiling untuk single Decision Tree telah tercapai.

## 2.5 Evaluasi Model Klasifikasi

Evaluasi model klasifikasi pada dataset tidak seimbang memerlukan strategi yang berbeda dari dataset seimbang. Accuracy bukan metrik yang tepat karena model dapat mencapai akurasi tinggi hanya dengan memprediksi semua sampel sebagai kelas mayoritas. Oleh karena itu, metrik seperti precision, recall, dan F1-Score untuk kelas minoritas lebih informatif [X].

Selain pemilihan metrik, strategi validasi juga menentukan keandalan hasil evaluasi. **Stratified split** membagi data secara acak dengan mempertahankan proporsi kelas yang sama di data latih dan data uji. Pendekatan ini cocok untuk eksplorasi model karena menyediakan cukup sampel minoritas di data latih. Namun, stratified split tidak merepresentasikan skenario deployment nyata di mana model harus memprediksi mahasiswa angkatan baru yang polanya mungkin berbeda dengan data historis.

**Temporal split** membagi data berdasarkan waktu: data latih berupa angkatan lama, data uji berupa angkatan baru. Pendekatan ini mensimulasikan kondisi deployment sesungguhnya dan menguji kemampuan generalisasi model terhadap distribution shift antar cohort [X]. Dalam penelitian ini, temporal split menggunakan angkatan <=2021 sebagai data latih (377 mahasiswa) dan angkatan >2021 sebagai data uji (231 mahasiswa). Kesenjangan hasil antara stratified split dan temporal split menjadi temuan kritis: F1(0)=0.889 pada stratified vs F1(0)=0.000 pada temporal.

**Repeated Stratified K-Fold** memberikan estimasi performa yang lebih stabil dengan mengulang k-fold cross-validation sebanyak beberapa kali [X]. Dalam penelitian ini, 10 kali ulang dengan 10 fold menghasilkan 100 evaluasi, memberikan confidence interval 95% untuk setiap metrik. Hasil CV menunjukkan bahwa stratified single split overestimates recall(0) sebesar +7.4% dibandingkan CV mean.

**Permutation importance** adalah metode alternatif untuk mengukur kepentingan fitur dengan mengacak nilai suatu fitur dan mengukur penurunan performa model [X]. Berbeda dengan MDI (Mean Decrease in Impurity) yang bias terhadap fitur kontinu dengan banyak unique values, permutation importance memberikan estimasi yang lebih tidak bias. Dalam penelitian ini, permutation importance mengkonfirmasi bahwa `sks_sem2` dan `sks_sem3` adalah dua fitur dominan, sementara 10 fitur lainnya memiliki kontribusi nol terhadap F1-Score pada model dengan depth=3.

## 2.6 Penanganan Class Imbalance

Class imbalance adalah kondisi di mana jumlah sampel antar kelas tidak proporsional. Dalam dataset penelitian ini, rasio kelas mayoritas terhadap minoritas adalah 88.8% berbanding 11.2% atau sekitar 8:1. Dampak class imbalance pada Decision Tree adalah kecenderungan algoritma untuk memilih split yang menguntungkan kelas mayoritas, menghasilkan pohon dengan daun yang didominasi kelas mayoritas [X].

Beberapa teknik penanganan class imbalance yang umum digunakan meliputi [X]:

**SMOTE (Synthetic Minority Oversampling Technique)** membuat sampel sintetis kelas minoritas dengan interpolasi antar sampel minoritas terdekat. SMOTE mengatasi kelemahan random oversampling yang hanya menduplikasi sampel eksisting. Namun, SMOTE dapat menciptakan sampel yang tidak realistis jika data memiliki noise.

**Undersampling** mengurangi jumlah sampel kelas mayoritas secara acak hingga seimbang dengan kelas minoritas. Teknik ini sederhana dan mengurangi waktu komputasi, namun berisiko membuang informasi berharga dari kelas mayoritas.

**class_weight** memberikan bobot lebih tinggi pada kelas minoritas saat menghitung fungsi loss [X]. Di scikit-learn, parameter `class_weight='balanced'` secara otomatis menetapkan bobot berbanding terbalik dengan frekuensi kelas. Pendekatan ini tidak mengubah data, hanya mengubah kontribusi setiap sampel terhadap fungsi loss.

Dalam penelitian ini, teknik class imbalance tidak diterapkan pada iterasi modeling utama. Keputusan ini didasarkan pada fokus awal untuk memahami arsitektur model dasar dan baseline Decision Tree tanpa modifikasi data. Eksperimen dengan `class_weight='balanced'` pada GridSearchCV menunjukkan bahwa bobot seimbang justru menyebabkan overfitting parah pada dataset dengan sampel minoritas yang sangat terbatas. Rekomendasi untuk iterasi lanjutan mencakup penerapan SMOTE pada temporal split untuk meningkatkan jumlah sampel minoritas di data latih dari 14 menjadi lebih representatif.

## 2.7 Data Preprocessing

Data preprocessing adalah tahap kritis dalam pipeline data mining yang bertujuan membersihkan dan mentransformasi data mentah menjadi format yang siap untuk modeling [X]. Dalam penelitian ini, preprocessing mencakup penanganan missing values, koreksi data, dan pembagian dataset.

**Handling missing values** dilakukan dalam dua langkah. Pertama, nilai IPS sebesar 0.0 diganti menjadi NaN karena teridentifikasi sebagai placeholder dari sistem legacy, bukan nilai akademik aktual. Sebanyak 219 mahasiswa (36%) memiliki setidaknya satu nilai IPS=0.0. Kedua, imputasi missing values menggunakan median global, bukan median per-angkatan. Pemilihan median global didasarkan pada temuan bahwa imputasi per-angkatan menciptakan proxy temporal: setiap angkatan mendapat nilai imputasi berbeda sehingga fitur seperti `sks_sem2` secara tidak sengaja dapat memisahkan angkatan, bukan memprediksi kelulusan.

**Perbedaan antara imputasi per-angkatan dan global median** memiliki dampak signifikan terhadap integritas temporal model. Imputasi per-angkatan menyebabkan data leakage karena informasi dari angkatan tertentu bocor ke nilai imputasi, membuat model belajar pola temporal yang tidak generalizable ke angkatan baru. Global median menghilangkan proxy ini dengan menggunakan satu nilai yang sama untuk seluruh dataset.

**Train-test split** dilakukan sebelum preprocessing untuk mencegah data leakage [X]. Dua pendekatan split digunakan: temporal split (angkatan <=2021 untuk train, >2021 untuk test) dan stratified split (80/20 acak dengan stratify=target). Fitur encoding tidak diperlukan karena seluruh fitur sudah numerik, dan scaling tidak diperlukan karena Decision Tree tidak sensitif terhadap skala fitur.

## 2.8 Variabel Penelitian

Variabel dalam penelitian ini dikelompokkan menjadi empat kategori:

**Variabel independen (14 prediktor)** merupakan fitur akademik yang digunakan untuk memprediksi ketepatan lulus, terdiri dari:
- Performa IPS: `ips_sem1`, `ips_sem2`, `ips_sem3`
- Beban SKS: `sks_sem1`, `sks_sem2`, `sks_sem3`
- Nilai matakuliah: `failed_courses`, `failed_in_sem1`, `repeated_courses`
- Fitur turunan: `ips_trend`, `avg_ips`, `ips_std`, `ips_min`, `sks_completion_ratio`

**Variabel dependen (target)** adalah status ketepatan lulus mahasiswa, yaitu `tepat_waktu` dengan nilai 1 untuk Tepat Waktu dan 0 untuk Tidak Tepat Waktu.

**Hyperparameter** yang dioptimalkan meliputi `max_depth`, `min_samples_leaf`, `criterion`, `splitter`, `max_features`, dan `min_impurity_decrease`.

**Variabel evaluasi** meliputi Accuracy, Precision, Recall, F1-Score, dan AUC-ROC dengan fokus utama pada Recall(0) dan F1(0) untuk kelas minoritas.

## 2.9 Tinjauan Jurnal Referensi

Tinjauan jurnal referensi dilakukan untuk memposisikan penelitian ini dalam konteks literatur yang ada, khususnya pada topik prediksi kelulusan mahasiswa menggunakan machine learning. Referensi yang dikumpulkan mencakup rentang tahun 2021 hingga 2026 dan membahas beberapa topik utama:

1. **Prediksi kelulusan dan performa akademik** menggunakan algoritma machine learning, termasuk Decision Tree, Random Forest, dan Gradient Boosting pada data pendidikan tinggi.

2. **Penerapan Decision Tree pada data pendidikan**, termasuk studi tentang identifikasi faktor-faktor yang memengaruhi kelulusan tepat waktu.

3. **CRISP-DM dalam studi kasus data mining**, khususnya implementasi CRISP-DM pada proyek prediksi di institusi pendidikan.

4. **Penanganan class imbalance pada data akademik**, termasuk teknik SMOTE, undersampling, dan class weight pada dataset dengan ketidakseimbangan kelas tinggi.

5. **Temporal validation dan distribution shift** dalam educational data mining, termasuk metodologi validasi yang sesuai untuk data dengan perubahan distribusi antar cohort.

(Daftar pustaka lengkap akan diisi oleh anggota kelompok yang bertugas mengumpulkan referensi jurnal 2021-2026.)

---

# Bab III - Metodologi Penelitian

## 3.1 Sumber Data

Data yang digunakan dalam penelitian ini berasal dari database akademik institusi perguruan tinggi yang diberi nama **LITIGASI**. Database dijalankan pada Microsoft SQL Server 2022 yang berjalan di atas container Docker dengan image `mcr.microsoft.com/mssql/server:2022-latest`. Database ini menyimpan seluruh data akademik institusi mulai dari data mahasiswa, nilai, kehadiran, hingga data perwalian dalam 405 tabel yang tersimpan di skema `dbo`.

Dari 405 tabel yang tersedia, hanya 4 tabel yang digunakan dalam penelitian ini setelah melalui proses eksplorasi dan profiling data:

1. **`tblMHS`** - Tabel master mahasiswa yang berisi data demografi dan status akademik, seperti Nomor Induk Mahasiswa (NIM), nama lengkap, angkatan, program studi, status mahasiswa (Lulus, Aktif, Keluar, Cuti), dan data demografi dasar. Tabel ini menjadi tabel utama untuk menghubungkan data dari tabel lain melalui kolom NIM.

2. **`IPSIPK`** - Tabel historis Indeks Prestasi Semester (IPS), Indeks Prestasi Kumulatif (IPK), dan Total SKS (TSKS) per semester untuk setiap mahasiswa. Tabel ini merupakan sumber utama fitur IPS dan SKS yang menjadi prediktor dalam penelitian. Setiap mahasiswa dapat memiliki banyak record sesuai dengan jumlah semester yang telah ditempuh.

3. **`Qnilai_mhs`** - Tabel nilai mata kuliah yang berisi data nilai setiap mata kuliah yang diambil oleh mahasiswa, termasuk nilai huruf (A, B, C, D, E) dan informasi kehadiran. Tabel ini digunakan untuk menghitung fitur turunan seperti jumlah mata kuliah gagal (failed courses), jumlah mata kuliah yang diulang (repeated courses), dan rasio penyelesaian SKS.

4. **`Kul_Kehadiran`** - Tabel kehadiran kuliah yang menyediakan data kehadiran sebagai data suplemen. Tabel ini dieksplorasi untuk menyusun fitur rata-rata kehadiran mahasiswa, namun akhirnya tidak digunakan dalam dataset final karena 53% nilai kehadiran hilang (missing).

Selain keempat tabel di atas, beberapa tabel lain turut dieksplorasi namun ditolak karena berbagai alasan:

- **`HtblNilai`** (39.990 baris) - Tabel nilai dari sistem legacy. Tidak ada satu pun NIM yang cocok dengan `tblMHS` (0% overlap) karena format NIM yang berbeda, sehingga tabel ini tidak dapat digunakan.

- **`Perwalian`** (6.222 baris) - Tabel perwalian yang berisi data SKS per semester. Seluruh kolom TSKSB (Total SKS) bernilai NULL, sehingga tidak dapat digunakan untuk mengekstrak informasi SKS.

- **`feed_nilai`** (1.457 baris) - Tabel nilai tambahan yang hanya mencakup 184 mahasiswa, tidak mencakup seluruh populasi.

- **`tblCuti`** (0 baris) - Tabel cuti yang kosong; informasi cuti mahasiswa sudah tercakup di kolom `Status` pada `tblMHS`.

- **`Luusan`** (0 baris) - Tabel lulusan yang kosong; informasi kelulusan sudah tersedia di `tblMHS.Status`.

Hasil akhir dari ekstraksi dan integrasi keempat tabel utama menghasilkan dataset final berupa 608 mahasiswa dengan 14 fitur prediktor dan 1 variabel target (`tepat_waktu`). Seluruh proses ekstraksi dan transformasi data dilakukan di memori menggunakan Python dengan pipeline `extract_dataset.py` yang membaca data dari SQL Server, melakukan join dan agregasi di sisi Python, kemudian menghasilkan file CSV sebagai output.

## 3.2 Proses Filtering Data (1.621 → 608)

Dataset awal dari tabel `tblMHS` mencatat sebanyak 1.621 mahasiswa dari dua program studi, yaitu AP (D3) sebanyak 386 mahasiswa dan IH (S1) sebanyak 1.235 mahasiswa. Dari jumlah tersebut, hanya 608 mahasiswa yang memenuhi kriteria untuk digunakan dalam pemodelan. Proses filtering dilakukan melalui tiga tahap eliminasi:

**Tahap 1 - Eliminasi mahasiswa dengan data IPSIPK kurang dari 3 semester (-458 mahasiswa).**

Mahasiswa angkatan 2024 dan 2025 dikeluarkan dari dataset karena belum memiliki data akademik minimal tiga semester yang disyaratkan sebagai fitur prediktor. Pada saat data diambil, mahasiswa angkatan 2024 baru menempuh 1-2 semester, sedangkan angkatan 2025 baru memasuki perkuliahan. Sebanyak 458 mahasiswa dieliminasi pada tahap ini karena tidak memenuhi threshold minimal tiga semester IPSIPK.

**Tahap 2 - Eliminasi mahasiswa dengan nol IPS valid (-154 mahasiswa).**

Sebanyak 154 mahasiswa dikeluarkan karena tidak memiliki satu pun nilai IPS yang valid pada data IPSIPK. Seluruh record IPS untuk mahasiswa ini bernilai NULL. Akar permasalahan ini adalah efek migrasi vendor database, di mana sistem akademik lama tidak mencatat data IPS per semester secara digital. Dampak migrasi ini paling parah dirasakan oleh angkatan 2014 yang 100% record IPSIPK-nya NULL, dan angkatan 2015 yang memiliki NULL rate 70-100%. Mahasiswa angkatan ini tetap tercatat di `tblMHS` karena data demografi dan status kelulusan berhasil dimigrasi, namun riwayat akademik per semesternya hilang.

**Tahap 3 - Eliminasi mahasiswa dengan outcome belum diketahui (-401 mahasiswa).**

Mahasiswa dengan status `Aktif` (612 mahasiswa) dan `Cuti` (41 mahasiswa) dikeluarkan dari dataset karena status ketepatan lulusnya belum dapat ditentukan - mereka masih dalam batas masa studi normal atau sedang menjalani cuti. Hanya mahasiswa dengan status `L` (Lulus, 302 mahasiswa), `Lulus` (177 mahasiswa), dan `Keluar` (155 mahasiswa) yang dipertahankan karena outcome ketepatan lulusnya sudah diketahui. Dari 155 mahasiswa dengan status `Keluar`, mereka tetap dimasukkan ke dalam dataset sebagai kelas "Tidak Tepat Waktu" (negatif) karena tidak menyelesaikan studi tepat waktu.

Setelah melalui ketiga tahap eliminasi, tersisa **608 mahasiswa** yang menjadi dataset final penelitian. Dataset ini terdiri dari mahasiswa dari berbagai angkatan (2015–2023) dengan dua program studi dan mencakup seluruh variabel yang diperlukan untuk pemodelan prediksi ketepatan lulus.

## 3.3 Preprocessing Data

### 3.3.1 Ekstraksi Dataset

Proses ekstraksi data dari basis data SQL Server hingga menjadi dataset siap olah dilakukan melalui pipeline `extract_dataset.py` (424 baris) yang dirancang untuk membaca data dari keempat tabel utama secara massal, memprosesnya di memori, dan menghasilkan file CSV. Pipeline ini terdiri dari enam langkah utama:

1. **Bulk query** - Keempat tabel (`IPSIPK`, `Qnilai_mhs`, `Kul_Kehadiran`, `tblMHS`) dibaca secara penuh dari SQL Server ke dalam memori Python menggunakan koneksi `pymssql`.
2. **Join dan agregasi** - Data dari keempat tabel digabungkan per mahasiswa menggunakan Python (bukan SQL JOIN) untuk fleksibilitas dalam penanganan anomali data.
3. **Filtering** - Mahasiswa dengan data tidak mencukupi atau outcome yang belum diketahui dikeluarkan sesuai tahapan filtering pada sub-bab 3.2.
4. **Perhitungan fitur turunan** - Fitur derived seperti `ips_trend`, `avg_ips`, `ips_std`, `ips_min`, dan `sks_completion_ratio` dihitung dari data IPS dan SKS.
5. **Penentuan target** - Label `tepat_waktu` ditentukan berdasarkan status kelulusan mahasiswa.
6. **Output** - Tiga file CSV dihasilkan: `dataset.csv` (full, 608 baris), `dataset_train.csv` (angkatan &lt;=2021, 377 baris), dan `dataset_test.csv` (angkatan &gt;2021, 231 baris).

**Perbaikan bug SKS kumulatif.** Salah satu temuan kritis pada fase data understanding adalah anomali pada kolom `TSKS` di tabel `IPSIPK`. Untuk mahasiswa angkatan 2015–2019, nilai `TSKS` merepresentasikan SKS per semester dengan rentang wajar 4–24. Namun untuk mahasiswa angkatan 2020+, nilai `TSKS` menyimpan total SKS kumulatif, bukan SKS per semester, dengan nilai mencapai 80–133 pada semester 1 atau 2. Skema penyimpanan yang berbeda antar angkatan ini merupakan inkonsistensi database yang diwarisi dari proses migrasi vendor.

Perbaikan dilakukan dengan deteksi otomatis: jika `TSKS` bernilai lebih dari 30 pada salah satu dari empat semester pertama, mahasiswa tersebut diflag sebagai abnormal. Untuk mahasiswa abnormal, nilai SKS diturunkan (derive) dari jumlah *distinct* `Kode_MK` di tabel `Qnilai_mhs` pada periode semester yang sesuai, dengan batas bawah 1 dan batas atas 20 mata kuliah per semester. Nilai SKS yang masih berada di luar rentang wajar setelah derivasi dikonversi menjadi NULL untuk diimputasi pada tahap preprocessing selanjutnya.

**Transformasi variabel turunan.** Setelah data IPS dan SKS terkumpul, lima fitur turunan dihitung untuk menangkap pola akademik yang tidak tercakup oleh nilai mentah per semester:
- **`ips_trend`** - Selisih `ips_sem3` dikurangi `ips_sem1`. Nilai positif menunjukkan tren membaik, negatif menunjukkan tren menurun.
- **`avg_ips`** - Rata-rata IPS dari tiga semester pertama, mencerminkan performa akademik secara umum.
- **`ips_std`** - Standar deviasi IPS antar semester, mengukur volatilitas atau konsistensi performa mahasiswa.
- **`ips_min`** - Nilai IPS terendah di antara tiga semester pertama, mengidentifikasi semester terburuk mahasiswa.
- **`sks_completion_ratio`** - Rasio total SKS yang diambil dalam tiga semester pertama dibagi 60 (asumsi SKS ideal untuk 2 tahun pertama pada program D3/S1). Fitur ini mengukur seberapa padat beban studi mahasiswa relatif terhadap kurikulum standar.

### 3.3.2 Data Cleansing

Dataset mentah hasil ekstraksi masih mengandung sejumlah masalah kualitas data yang memerlukan pembersihan sebelum dapat digunakan untuk pemodelan. Proses data cleansing dilakukan dalam tiga langkah utama: koreksi placeholder sistem, imputasi missing values, dan eliminasi fitur yang tidak layak.

**Koreksi zero-IPS placeholder.** Sebanyak 219 mahasiswa (36% dari total dataset) memiliki setidaknya satu nilai IPS sebesar 0.0 pada data IPSIPK. Investigasi menunjukkan bahwa 79% dari mahasiswa dengan IPS=0.0 tetap lulus tepat waktu, mengonfirmasi bahwa nilai tersebut bukan merupakan nilai akademik aktual melainkan placeholder dari sistem legacy. Sistem lama tidak mencatat nilai IPS per semester dan menggunakan 0.0 sebagai nilai default. Seluruh nilai IPS=0.0 dikonversi menjadi `NaN` untuk ditangani pada tahap imputasi, sehingga tidak diinterpretasikan oleh model sebagai performa akademik nol.

**Imputasi missing values dengan global median.** Setelah koreksi placeholder, missing values terdapat pada fitur IPS dan SKS di tiga semester pertama. Penanganan missing values mempertimbangkan dua pendekatan:

*Pendekatan awal - imputasi per-angkatan.* Median dihitung terpisah untuk setiap angkatan. Pendekatan ini awalnya dipilih karena distribusi IPS dan SKS berbeda antar angkatan. Namun, pada iterasi awal modeling ditemukan bahwa imputasi per-angkatan menciptakan *temporal proxy*: setiap angkatan mendapat nilai imputasi yang berbeda, sehingga fitur seperti `sks_sem2` secara tidak sengaja dapat memisahkan angkatan satu dengan lainnya. Model belajar pola temporal ("angkatan A punya SKS lebih tinggi dari angkatan B") alih-alih pola akademik yang sesungguhnya. Akibatnya, performa model pada stratified split tinggi (F1(0)=0,867) namun pada temporal split rendah (F1(0)=0,060) - model tidak dapat menggeneralisasi ke angkatan baru karena proxy temporal ini tidak ada pada data uji.

*Pendekatan final - imputasi global median.* Untuk menghilangkan temporal proxy, imputasi dilakukan menggunakan satu nilai median yang dihitung dari seluruh dataset (global median) untuk setiap fitur. Dengan pendekatan ini, mahasiswa dari angkatan mana pun yang memiliki missing values mendapat nilai imputasi yang sama, sehingga tidak ada informasi temporal yang bocor ke dalam fitur. Global median imputation menghasilkan nilai yang lebih konservatif namun secara temporal lebih fair.

**Eliminasi fitur leakage dan non-prediktif.** Sejumlah fitur dikeluarkan dari dataset karena termasuk dalam salah satu kategori berikut:

*Fitur leakage* - fitur yang secara tidak sah memberikan informasi tentang target karena hubungan kausal terbalik:
- **`ips_sem4` dan `sks_sem4`** - Data semester 4 tidak boleh digunakan sebagai prediktor karena hanya mahasiswa yang masih aktif pada semester 4 yang memiliki data ini. Korelasi `ips_sem4` dengan target mencapai r=0,877, mengindikasikan leakage signifikan.
- **`ipk_sem4`** - Redundant dengan `avg_ips` dan juga bersifat leakage karena membutuhkan data hingga semester 4.
- **`semester_count`** - Circular dengan target; jumlah semester terdaftar secara langsung menentukan apakah seorang mahasiswa dikategorikan tepat waktu atau tidak.
- **`angkatan`** - Tahun masuk mahasiswa berkorelasi dengan kebijakan akademik dan sistem pencatatan data, bukan dengan performa individu. Penggunaan `angkatan` sebagai fitur juga berisiko menciptakan temporal proxy yang serupa dengan masalah imputasi per-angkatan.

*Fitur non-prediktif* - fitur dengan daya pisah rendah terhadap kelas target:
- **`avg_attendance` dan `has_attendance`** - Rata-rata kehadiran memiliki 53% missing values dan korelasi mendekati nol terhadap target (r=-0,016). Fitur ini di-drop karena informasi yang tersedia tidak cukup untuk menjadi prediktor yang andal.
- **`ips_max`** - Nilai IPS tertinggi memiliki korelasi hampir nol (r≈0) dan tidak membedakan kedua kelas.
- **`id_agama`** - Hanya 4 dari 608 mahasiswa beragama Hindu, sisanya Islam; korelasi dengan target sangat rendah (r=0,031).
- **`jenis_kelamin`** - Korelasi dengan target hanya r=0,050, tidak cukup untuk membantu klasifikasi.
- **`program`** - Program studi (D3/S1) memiliki near-zero importance pada eksperimen awal. Meskipun distribusi kelas negatif berbeda antar program (D3=6,8%, S1=12,6%), model Decision Tree tidak memanfaatkan fitur ini dalam pemisahan kelas.
- **`total_sks_lulus_sem4`** - Redundant dengan `sks_completion_ratio` yang sudah mencakup tiga semester pertama.

Setelah proses cleansing, dataset berkurang dari 27 kolom mentah menjadi 16 fitur prediktor plus 1 target (`tepat_waktu`), seluruhnya bertipe numerik tanpa missing values.

### 3.3.3 Train-Test Split

Pembagian dataset menjadi data latih (train) dan data uji (test) merupakan langkah krusial sebelum pemodelan untuk memastikan evaluasi yang objektif terhadap kemampuan generalisasi model. Dalam penelitian ini, dua pendekatan split digunakan secara paralel untuk mengukur kesenjangan performa antara kondisi eksperimental dan skenario deployment nyata.

**Temporal split** membagi data berdasarkan waktu (angkatan) untuk mensimulasikan kondisi deployment sesungguhnya. Model dilatih hanya pada data mahasiswa angkatan lama yang outcome-nya sudah diketahui, kemudian diuji pada mahasiswa angkatan baru yang belum pernah dilihat model. Detail pembagian:

| Split | Angkatan | Jumlah | Tepat | Tidak | % Negatif |
|-------|----------|-------:|------:|------:|----------:|
| Train | &lt;= 2021 | 377 | 363 | 14 | 3,7% |
| Test | &gt; 2021 | 231 | 177 | 54 | 23,4% |

Temporal split dipilih sebagai pendekatan utama karena tiga alasan. Pertama, skenario ini mencerminkan penggunaan model di dunia nyata, di mana model dilatih pada data historis dan digunakan untuk memprediksi mahasiswa angkatan baru. Kedua, pembagian berdasarkan waktu menghindari data leakage yang mungkin terjadi pada random split, seperti mencampur mahasiswa seangkatan yang memiliki karakteristik serupa ke dalam train dan test. Ketiga, temporal split menguji ketahanan model terhadap distribution shift antar cohort, yaitu perubahan distribusi fitur yang terjadi seiring waktu.

**Stratified split** membagi data secara acak dengan proporsi 80/20 sambil mempertahankan distribusi kelas yang sama di kedua bagian. Detail pembagian:

| Split | Jumlah | Tepat | Tidak | % Negatif |
|-------|-------:|------:|------:|----------:|
| Train | 486 | 432 | 54 | 11,1% |
| Test | 122 | 108 | 14 | 11,5% |

Stratified split digunakan sebagai pembanding untuk mengukur efektivitas temporal split. Pada stratified split, sampel kelas minoritas yang tersedia di data latih jauh lebih banyak (54 negatif) dibandingkan temporal split (14 negatif), sehingga model memiliki lebih banyak data untuk mempelajari pola kelas minoritas. Namun, stratified split tidak merepresentasikan skenario deployment nyata karena mencampur angkatan lama dan baru secara acak.

Perbedaan mendasar antara kedua pendekatan ini menjadi fokus evaluasi penelitian. Temporal split menguji apakah model yang dilatih pada data historis dapat menggeneralisasi ke mahasiswa baru, sementara stratified split menguji performa model dalam kondisi optimal. Kesenjangan hasil antara keduanya menjadi temuan kritis yang akan dibahas pada bab hasil dan pembahasan.

## 3.4 Iterasi Pemodelan

Pemodelan dilakukan dalam tiga iterasi bertahap dengan pendekatan CRISP-DM yang iteratif. Setiap iterasi dibangun berdasarkan temuan dari iterasi sebelumnya untuk secara sistematis mengatasi masalah yang teridentifikasi.

### 3.4.1 Iterasi 1 - Baseline

Iterasi pertama bertujuan membangun model Decision Tree awal dengan hyperparameter default dari scikit-learn untuk memperoleh gambaran dasar mengenai kemampuan algoritma dalam mengklasifikasikan ketepatan lulus. Tiga eksperimen dilakukan secara paralel untuk mengidentifikasi masalah mendasar sebelum memasuki tahap tuning.

**Eksperimen 01 - Temporal baseline.** Model Decision Tree dengan parameter default (`max_depth=None`, `min_samples_leaf=1`, `criterion='gini'`) dilatih pada temporal split: angkatan <=2021 sebagai data latih (377 mahasiswa, 14 negatif) dan diuji pada angkatan >2021 (231 mahasiswa, 54 negatif). Hasilnya menunjukkan kegagalan total: Recall(0)=0,037 - hanya 2 dari 54 mahasiswa berisiko berhasil terdeteksi. Precision(0)=1,000 (model sangat konservatif, hanya membuat 2 prediksi negatif dan keduanya benar), namun F1(0)=0,071 dan AUC=0,519 (hampir setara random). Confusion matrix mengungkapkan 52 false negative dari 54 mahasiswa berisiko.

Akar masalahnya adalah ketimpangan jumlah sampel minoritas di data latih. Dengan hanya 14 sampel negatif (3,7%), Decision Tree tidak memiliki cukup data untuk mempelajari pola kelas minoritas secara general. Tree yang dihasilkan memiliki kedalaman 9 dengan 21 daun - 5 fitur memiliki zero importance, dan struktur tree didominasi oleh split yang menguntungkan kelas mayoritas. Analisis root split menunjukkan bahwa hanya 1,6% data mengalir ke cabang yang menghasilkan prediksi kelas minoritas.

**Eksperimen 02 - Stratified baseline.** Untuk menguji apakah penambahan sampel minoritas di data latih memperbaiki recall, eksperimen kedua menggunakan stratified random split 80/20 dengan `stratify=target`. Hasilnya, data latih memiliki 54 sampel negatif (11,1%) - hampir 4 kali lipat dari temporal split. Selain itu, fitur `angkatan` dan `program` di-drop untuk menghindari data leakage (`angkatan` menyebabkan tree langsung mempelajari shortcut bahwa angkatan 2023 seluruhnya negatif karena mahasiswa belum lulus).

Hasil stratified baseline menunjukkan perbaikan dramatis: Recall(0)=0,929 (13 dari 14 mahasiswa berisiko terdeteksi), F1(0)=0,765, dan AUC=0,932. Namun performa tinggi ini perlu diinterpretasi dengan hati-hati. Tree memiliki kedalaman 8 dengan train accuracy=1,0 - model menghafal seluruh data latih (overfitting). Feature importance didominasi oleh `sks_sem2` dengan 0,621 - root split berada di `sks_sem2 <= 18,50`, dan cabang kanan threshold ini mengandung 63 dari 68 mahasiswa tidak tepat waktu (negatif rate 53,7%). Precision(0)=0,650 menunjukkan bahwa model juga menghasilkan false positive yang cukup banyak.

**Eksperimen 03 - Investigasi SKS.** Dominasi `sks_sem2` pada stratified baseline (61,2% importance) menimbulkan pertanyaan: apakah ini sinyal genuine atau artifact dari bug SKS kumulatif yang mungkin belum tuntas diperbaiki? Eksperimen investigasi dilakukan dengan menganalisis distribusi `sks_sem2` secara mendalam.

Hasil investigasi mengkonfirmasi bahwa dominasi `sks_sem2` adalah **sinyal genuine**, bukan artifact data. Mahasiswa tidak tepat waktu memiliki rata-rata SKS semester 2 sebesar 19,43, sedangkan mahasiswa tepat waktu memiliki rata-rata 12,96 - perbedaan yang jelas (selisih 6,47 SKS). Korelasi Pearson `sks_sem2` dengan target mencapai -0,348. Tidak ditemukan nilai abnormal (>24 atau <2) pada dataset bersih.

Namun, investigasi lebih lanjut mengungkapkan masalah yang lebih substansial. Distribusi `sks_sem2` per angkatan menunjukkan pola yang sangat bervariasi: angkatan 2017 semuanya memiliki `sks_sem2=22` (seragam), angkatan 2022 mayoritas memiliki `sks_sem2=9`, sementara angkatan 2023 mayoritas memiliki `sks_sem2=19`. Variasi antar angkatan ini tidak sepenuhnya natural - sebagian disebabkan oleh imputasi median per-angkatan yang diterapkan pada preprocessing. Imputasi per-angkatan memberikan nilai imputasi yang berbeda untuk setiap angkatan, sehingga fitur `sks_sem2` secara tidak sengaja bertindak sebagai proxy temporal yang membedakan angkatan, bukan sebagai prediktor akademik murni.

**Temuan kritis Iterasi 1:** Terdapat dua masalah fundamental yang perlu diatasi. Pertama, **temporal split tidak layak** dengan hanya 14 sampel negatif di data latih - model tidak memiliki cukup informasi untuk belajar. Kedua, **imputasi per-angkatan menciptakan temporal proxy** - nilai imputasi yang berbeda per angkatan membuat fitur SKS secara artifisial berkorelasi dengan angkatan, menyebabkan model belajar pola temporal yang tidak generalizable. Kedua temuan ini menjadi dasar untuk Iterasi 2.

### 3.4.2 Iterasi 2 - Global Median Imputation

Iterasi kedua bertujuan menghilangkan temporal proxy yang diidentifikasi pada Iterasi 1 dengan mengganti strategi imputasi dari median per-angkatan menjadi global median. Perubahan ini hanya mengubah satu aspek preprocessing: seluruh missing values pada fitur IPS dan SKS diisi dengan satu nilai median yang dihitung dari seluruh dataset, bukan dari masing-masing angkatan.

**Perubahan imputasi.** Sebelumnya, imputasi dilakukan per-angkatan menggunakan kode `df[col] = df.groupby('angkatan')[col].transform(lambda x: x.fillna(x.median()))`. Pendekatan ini memberikan nilai imputasi yang berbeda untuk setiap angkatan - angkatan 2015 mendapat nilai imputasi yang dipengaruhi oleh pola IPS/SKS angkatan 2015, angkatan 2022 mendapat nilai yang dipengaruhi oleh pola angkatan 2022, dan seterusnya. Akibatnya, fitur seperti `sks_sem2` secara tidak sengaja mengandung informasi temporal yang memungkinkan tree membedakan angkatan.

Setelah perbaikan, imputasi menggunakan satu nilai global: `df[col] = df[col].fillna(df[col].median())`. Seluruh mahasiswa dengan missing values mendapat nilai imputasi yang sama, sehingga tidak ada informasi temporal yang bocor ke dalam fitur. Sebanyak 47,5% baris dataset memiliki nilai `sks_sem2` yang diimputasi - seluruhnya mendapat nilai global median 18,0.

**Hasil temporal split dengan global median.** Temporal split tetap menunjukkan kegagalan: Recall(0)=0,056, F1(0)=0,105, AUC=0,528. Hasil ini mengkonfirmasi bahwa akar masalah temporal split bukan pada strategi imputasi, melainkan pada jumlah sampel minoritas yang tidak mencukupi di data latih (14 sampel). Global median tidak dapat memperbaiki fundamental imbalance ini.

**Hasil stratified split dengan global median.** Hasil stratified split justru menunjukkan peningkatan di semua metrik dibandingkan Iterasi 1:

| Metrik | Per-Angkatan (Iterasi 1) | Global Median (Iterasi 2) | Perubahan |
|--------|:------------------------:|:------------------------:|:---------:|
| Recall(0) | 0,929 | 0,929 | = |
| Precision(0) | 0,650 | **0,813** | +0,163 |
| F1(0) | 0,765 | **0,867** | +0,102 |
| AUC | 0,932 | **0,950** | +0,018 |
| sks_sem2 importance | 0,621 | **0,432** | -0,189 |
| sks_sem3 importance | 0,095 | **0,280** | +0,185 |

Peningkatan precision dari 0,650 menjadi 0,813 berarti jumlah false positive berkurang dari 7 menjadi hanya 2 mahasiswa. Confusion matrix menunjukkan 13 dari 14 mahasiswa berisiko terdeteksi dengan hanya 1 false negative, dan hanya 2 false positive dari 108 mahasiswa tepat waktu.

**Feature importance lebih seimbang.** Dampak paling signifikan dari global median adalah distribusi feature importance yang lebih merata. Dominasi `sks_sem2` turun dari 0,621 menjadi 0,432, sementara `sks_sem3` naik dari 0,095 menjadi 0,280. Perubahan ini menunjukkan bahwa imputasi per-angkatan sebelumnya telah mengontaminasi fitur SKS dengan sinyal temporal palsu - `sks_sem2` menyerap sebagian besar informasi karena nilai imputasinya bervariasi antar angkatan. Setelah kontaminasi dihilangkan, bobot importance terdistribusi lebih wajar antara `sks_sem2` dan `sks_sem3`, mencerminkan bahwa kedua fitur SKS bersama-sama menangkap pola beban studi mahasiswa.

Root split tetap berada di `sks_sem2 <= 18,50` dengan struktur tree yang mirip dengan Iterasi 1. Aturan bisnis utama yang teridentifikasi meliputi: (1) `sks_sem2 <= 18,5` dan `ips_sem1 <= 2,70` diprediksi tidak tepat - mahasiswa dengan SKS normal namun IPS semester 1 rendah; (2) `sks_sem2 > 18,5` dan `sks_sem3 > 18,5` dan `ips_sem1 <= 2,93` diprediksi tidak tepat - mahasiswa overload SKS di dua semester awal dengan performa awal buruk; (3) `sks_sem2 > 18,5` dan `ips_std > 0,27` diprediksi tidak tepat - SKS tinggi dengan volatilitas IPS tinggi.

**Kesimpulan Iterasi 2.** Global median imputation berhasil menghilangkan temporal proxy tanpa mengorbankan performa - justru meningkatkan precision secara signifikan. Model stratified + global median menjadi baseline terbaik dengan F1(0)=0,867, melampaui seluruh target modeling plan. Namun, masalah overfitting masih belum terselesaikan: tree memiliki kedalaman 8 dengan 24 daun dan train accuracy sempurna 1,0. Penanganan overfitting melalui hyperparameter tuning menjadi fokus Iterasi 3.

### 3.4.3 Iterasi 3 - Hyperparameter Tuning

Iterasi ketiga bertujuan mengatasi overfitting pada model baseline dengan melakukan hyperparameter tuning secara sistematis menggunakan GridSearchCV. Tiga eksperimen dilakukan untuk menemukan konfigurasi hyperparameter optimal yang menghasilkan model sederhana namun tetap memiliki performa tinggi.

**Eksperimen 01 - Pre-Pruning GridSearchCV.** Pencarian kombinasi hyperparameter dilakukan pada grid seluas 240 kombinasi dengan 5-fold cross-validation, menghasilkan total 1.200 fit. Hyperparameter yang dioptimalkan meliputi `max_depth` (3, 4, 5, 6, None), `min_samples_leaf` (5, 10, 15, 20), `min_samples_split` (5, 10, 20), `class_weight` (None, balanced), dan `criterion` (gini, entropy). Scoring menggunakan F1-Score kelas minoritas (F1(0)) untuk memastikan optimasi berfokus pada deteksi mahasiswa berisiko.

Konfigurasi terbaik ditemukan pada `max_depth=3`, `min_samples_leaf=10`, dan `criterion='gini'` dengan CV F1(0)=0,8165. Parameter `class_weight=None` terpilih karena `class_weight='balanced'` justru menyebabkan overfitting lebih parah pada dataset dengan sampel minoritas yang sangat terbatas. Parameter `min_samples_split` dan `splitter` tidak berpengaruh signifikan pada kompleksitas pohon.

Model terbaik memiliki kedalaman 3 dengan 7 daun dan 13 node - reduksi 62,5% depth, 70,8% leaves, dan 72,3% nodes dibandingkan baseline Iterasi 2. Overfitting berhasil dihilangkan: train accuracy turun dari 1,0000 menjadi 0,9691 tanpa peningkatan signifikan pada bias. Performa test: F1(0)=0,889, Recall(0)=0,857, Precision(0)=0,923, AUC=0,924. Confusion matrix menunjukkan 12 dari 14 mahasiswa berisiko terdeteksi dengan hanya 2 false positive.

Feature importance model terakhir menunjukkan dominasi fitur SKS yang lebih ekstrem dibandingkan Iterasi 2. `sks_sem2` memiliki importance 0,558 dan `sks_sem3` memiliki 0,361 - keduanya menyumbang 91,9% dari total importance. Hanya 6 dari 14 fitur yang digunakan tree, dan 10 fitur sisanya (termasuk seluruh fitur IPS dan nilai mata kuliah) memiliki kontribusi nol terhadap prediksi. Root split tetap di `sks_sem2 <= 18,50`, persis seperti Iterasi 1 dan 2, menunjukkan stabilitas aturan keputusan.

**Eksperimen 02 - Feature Engineering (dihapus).** Empat fitur baru ditambahkan ke dataset: `sks_sem2_high` (binary flag SKS semester 2 di atas rata-rata), `sks_sem3_high` (binary flag SKS semester 3 di atas rata-rata), `sks_total` (total SKS tiga semester), dan `ips_sem1_low` (binary flag IPS semester 1 di bawah rata-rata). Seluruh fitur baru memiliki feature importance 0,000 - tree mengabaikan binary flags karena sudah menemukan threshold optimal sendiri pada fitur kontinu yang ada. Semua fitur engineering dihapus dan tidak digunakan pada eksperimen selanjutnya.

**Eksperimen 03 - Combined Tuning.** Eksperimen ketiga menambahkan dua hyperparameter yang belum dioptimalkan sebelumnya: `max_features` (None, sqrt, 4, 5) dan `min_impurity_decrease` (0,0, 0,001, 0,002, 0,005, 0,008, 0,01, 0,02), dengan `max_depth` difixed di 7 dan `min_samples_leaf` di 10. Total 28 kombinasi dengan 5-fold cross-validation (140 fit).

Konfigurasi terbaik: `max_features=5` dan `min_impurity_decrease=0,001` dengan CV F1(0)=0,8371 (naik dari 0,8165). Analisis impurity landscape pada tree tanpa constraint menunjukkan bahwa 8 dari 12 internal nodes (67%) adalah cosmetic splits - split di mana kedua anak menghasilkan kelas yang sama. Threshold `min_impurity_decrease=0,001` menghilangkan 5 cosmetic splits, hanya menyisakan 7 split yang benar-benar informatif.

Namun, performa test tetap stagnan pada F1(0)=0,889 - identik dengan Eksperimen 01. Temuan ini mengindikasikan bahwa ceiling untuk single Decision Tree telah tercapai pada dataset ini. Tidak ada kombinasi hyperparameter tambahan yang dapat meningkatkan performa test di luar F1(0)=0,889 dengan algoritma Decision Tree standar.

**Model terbaik.** Model final dipilih dari Eksperimen 01 (Pre-Pruning) karena memiliki kompleksitas terendah dengan performa tertinggi:

| Properti | Baseline (Iterasi 2) | Model Final (Iterasi 3) |
|----------|:-------------------:|:-----------------------:|
| Kedalaman | 8 | **3** |
| Daun | 24 | **7** |
| Node | 47 | **13** |
| Fitur digunakan | 12/14 | **6/14** |
| Train Accuracy | 1,0000 | **0,9691** |
| F1(0) | 0,867 | **0,889** |
| Recall(0) | 0,929 | 0,857 |
| Precision(0) | 0,813 | **0,923** |

Model final menggunakan hanya 6 dari 14 fitur: `sks_sem2` (55,8%), `sks_sem3` (36,1%), `ips_std` (4,7%), `avg_ips` (1,7%), `ips_sem1` (1,0%), dan `sks_sem1` (0,7%). Aturan keputusan utama yang dihasilkan: **jika SKS semester 2 lebih dari 18,5 dan SKS semester 3 kurang dari atau sama dengan 18,5, maka mahasiswa diprediksi tidak lulus tepat waktu** - pola "overload lalu collapse" yang menjadi sinyal paling kuat dalam dataset.

## 3.5 Evaluasi Model

Setelah model final diperoleh dari Iterasi 3, evaluasi dilakukan secara komprehensif menggunakan enam fase analisis. Fokus utama evaluasi adalah membandingkan performa model pada stratified split (kondisi eksperimental) dengan temporal split (skenario deployment nyata), serta memvalidasi stabilitas dan interpretabilitas model.

### 3.5.1 Fase 1 - Temporal Validation (Temuan Kritis)

Model terbaik dari Iterasi 3 (`max_depth=3`, `min_samples_leaf=10`, `criterion='gini'`) dilatih ulang menggunakan temporal split dengan konfigurasi hyperparameter yang persis sama. Data latih terdiri dari angkatan <=2021 (377 mahasiswa, 14 negatif) dan data uji dari angkatan >2021 (231 mahasiswa, 54 negatif). Hasilnya menunjukkan kegagalan total:

| Metrik | Stratified Split | Temporal Split | Delta |
|--------|:----------------:|:--------------:|:-----:|
| Accuracy | 0,9754 | 0,7662 | -0,2092 |
| Precision(0) | 0,9231 | **0,0000** | -0,9231 |
| Recall(0) | 0,8571 | **0,0000** | -0,8571 |
| F1(0) | 0,8889 | **0,0000** | -0,8889 |
| AUC | 0,9550 | 0,7098 | -0,2452 |

Model memprediksi **seluruh 231 mahasiswa** sebagai "Tepat Waktu". Tidak ada satu pun dari 54 mahasiswa berisiko yang berhasil terdeteksi. Confusion matrix menunjukkan seluruh 54 mahasiswa tidak tepat waktu berada di kuadran false negative, sementara 177 mahasiswa tepat waktu diklasifikasikan dengan benar (true positive). Precision(0)=0,000 dan Recall(0)=0,000 mengonfirmasi bahwa model tidak memiliki kemampuan deteksi sama sekali pada skenario temporal.

**Analisis akar masalah.** Struktur tree temporal berbeda total dengan tree stratified. Pada stratified split, root split berada di `sks_sem2 <= 18,50` dengan dua daun yang memprediksi kelas negatif pada cabang `sks_sem2 > 18,50` dan `sks_sem3 <= 18,50`. Pada temporal split, root split bergeser ke `failed_courses <= 1,50` dan seluruh 7 daun memprediksi kelas positif (Tepat Waktu) - tidak ada satu pun daun yang memprediksi kelas minoritas.

Penyebab utamanya adalah interaksi antara jumlah sampel minoritas yang sangat terbatas dan constraint `min_samples_leaf=10`. Temporal train hanya memiliki 14 sampel negatif (3,7%) dari total 377 mahasiswa. Dengan `min_samples_leaf=10`, setiap daun harus memiliki minimal 10 sampel. Jika tree ingin membentuk daun yang memprediksi "Tidak Tepat", diperlukan minimal 10 sampel negatif di satu daun - tetapi dengan hanya 14 sampel negatif total, tree tidak dapat mengalokasikan 10 di antaranya ke satu daun sambil tetap menyisakan sampel untuk split lain yang informatif. Akibatnya, algoritma memilih untuk mengabaikan kelas minoritas sama sekali dan memprediksi semua sampel sebagai "Tepat Waktu" - strategi yang menghasilkan akurasi 76,6% tanpa mendeteksi satu pun mahasiswa berisiko.

Temuan ini sangat kritis karena mengungkapkan bahwa **performa tinggi pada stratified split (F1(0)=0,889) bersifat ilusif**. Stratified split menyediakan 54 sampel negatif di data latih dengan mencampur angkatan lama dan baru secara acak, sehingga model memiliki cukup data untuk mempelajari pola kelas minoritas. Namun, skenario ini tidak merepresentasikan kondisi deployment nyata. Pada kenyataannya, model hanya akan memiliki 14 sampel negatif historis untuk dilatih - jumlah yang tidak mencukupi untuk Decision Tree dengan `min_samples_leaf=10`.

### 3.5.2 Fase 2 - Deep Error Analysis

Analisis kesalahan dilakukan untuk memahami profil mahasiswa yang tidak terdeteksi oleh model temporal. Dengan model yang memprediksi seluruh 231 mahasiswa sebagai "Tepat Waktu", seluruh 54 mahasiswa berisiko menjadi false negative (100% miss rate), sementara tidak ada false positive karena model tidak pernah memprediksi kelas negatif.

**Profil false negative.** Seluruh 54 false negative memiliki karakteristik yang terkelompok:

| Karakteristik | Jumlah | Persentase |
|---------------|-------:|-----------:|
| Program IH (S1) | 52 | 96,3% |
| Program AP (D3) | 2 | 3,7% |
| Angkatan 2022 | 5 | 9,3% |
| Angkatan 2023 | 49 | 90,7% |
| Mean sks_sem2 | 19,67 | - |
| Mean sks_sem3 | 18,15 | - |
| Mean avg_ips | 3,08 | - |
| Mean failed_courses | 3,09 | - |

Mayoritas false negative berasal dari program sarjana (IH) sebesar 96,3%, sementara hanya 3,7% dari program vokasi (AP). Distribusi ini proporsional dengan komposisi dataset (IH 81% dari total), namun mengindikasikan bahwa kegagalan model terjadi secara merata di kedua program tanpa bias spesifik. Angkatan 2023 mendominasi false negative dengan 90,7% - hampir seluruh mahasiswa angkatan 2023 (49 dari 50) adalah tidak tepat waktu dan tidak terdeteksi oleh model.

**Analisis rule coverage.** Rule "overload-collapse" yang diperoleh dari model stratified diuji pada temporal test. Rule ini mendefinisikan mahasiswa berisiko sebagai mereka yang memiliki `sks_sem2 > 18,5` dan `sks_sem3 <= 18,5`. Hasil analisis menunjukkan:

| Rule | Coverage | Actual Negatif | Prediksi Model Temporal |
|------|:--------:|:--------------:|:----------------------:|
| sks_sem2 <= 18,5 | 177 | 0 | 0 (Tepat) |
| sks_sem2 > 18,5, sks_sem3 <= 18,5 | **52** | **52** | **0 (Tepat)** |
| sks_sem2 > 18,5, sks_sem3 > 18,5 | 2 | 2 | 0 (Tepat) |

Temuan kritis: rule "overload-collapse" mencakup 52 dari 54 mahasiswa berisiko di temporal test, dan **100% (52/52) adalah actual negatif** - precision rule ini sempurna. Namun, model temporal tidak memiliki satu pun leaf yang memprediksi kelas negatif, sehingga recall rule ini 0,000 - tidak ada satu mahasiswa pun yang terdeteksi. Rule ini sebenarnya telah teridentifikasi di model stratified dengan root split yang sama (`sks_sem2 <= 18,50`), namun model temporal tidak pernah mempelajarinya karena keterbatasan sampel minoritas.

**Distribution shift.** Analisis distribusi fitur antar angkatan mengungkapkan pergeseran signifikan yang menjadi faktor kedua kegagalan model temporal:

| Angkatan | sks_sem2 mean | sks_sem3 mean | avg_ips mean | Neg/Total |
|----------|:-------------:|:-------------:|:------------:|:---------:|
| 2020 | 10,47 | 8,25 | 3,18 | 4/40 |
| 2021 | 11,37 | 9,43 | 3,11 | 3/46 |
| 2022 | 17,64 | 9,10 | 2,71 | 5/181 |
| 2023 | 19,20 | 17,78 | 2,93 | 49/50 |

Mean `sks_sem2` meningkat drastis dari ~11 (angkatan 2021) menjadi ~19 (angkatan 2023) - lonjakan hampir 2 kali lipat. Mean `sks_sem3` juga naik dari ~9 (2022) menjadi ~18 (2023). Pergeseran distribusi ini berarti bahwa pola akademik mahasiswa angkatan baru (2022-2023) berbeda secara fundamental dari pola angkatan lama (2015-2021) yang digunakan sebagai data latih. Model yang dilatih pada data historis dengan SKS rendah tidak dapat menggeneralisasi ke cohort baru dengan SKS tinggi.

Tiga faktor penyebab kegagalan temporal validation saling terkait: (1) **sample imbalance** - hanya 14 sampel negatif di training tidak cukup untuk membentuk leaf prediksi minoritas; (2) **distribution shift** - perubahan pola SKS antar cohort membuat model historis tidak relevan; (3) **constraint hyperparameter** - `min_samples_leaf=10` terlalu restriktif untuk dataset dengan hanya 3,7% kelas minoritas.

### 3.5.3 Fase 3 - Repeated Stratified K-Fold Cross-Validation

Untuk mendapatkan estimasi performa model yang lebih stabil, Repeated Stratified K-Fold Cross-Validation dilakukan dengan 10 kali ulang dan 10 fold, menghasilkan total 100 evaluasi. Setiap evaluasi menggunakan model dengan konfigurasi yang sama (`max_depth=3`, `min_samples_leaf=10`) pada data latih stratified yang berbeda-beda.

Hasil cross-validation menunjukkan estimasi performa yang lebih konservatif dibandingkan single stratified split:

| Metrik | Mean | Std | CI 95% Bawah | CI 95% Atas |
|--------|:----:|:---:|:------------:|:-----------:|
| F1(0) | 0,833 | 0,096 | 0,814 | 0,851 |
| Recall(0) | 0,783 | 0,135 | 0,757 | 0,810 |
| Precision(0) | 0,911 | 0,103 | 0,890 | 0,931 |
| AUC | 0,970 | 0,040 | 0,962 | 0,978 |
| Accuracy | 0,966 | 0,018 | 0,962 | 0,970 |

Perbandingan antara single stratified split dengan CV mean mengungkapkan adanya overestimasi:

| Metrik | Single Split | CV Mean | Selisih | Status |
|--------|:-----------:|:-------:|:-------:|:------:|
| Accuracy | 0,9754 | 0,9660 | +0,0095 | Sesuai |
| Precision(0) | 0,9231 | 0,9105 | +0,0126 | Sesuai |
| Recall(0) | **0,8571** | **0,7833** | **+0,0738** | **Optimistik** |
| F1(0) | **0,8889** | **0,8327** | **+0,0562** | **Optimistik** |
| AUC | 0,9550 | 0,9700 | -0,0150 | Sesuai |

Recall(0) pada single stratified split overestimates sebesar **+7,4%** dibandingkan CV mean (0,783). F1(0) overestimates sebesar **+5,6%**. Overestimasi ini terjadi karena single split hanya menguji model pada satu komposisi train-test tertentu dengan 14 sampel negatif di test, sehingga hasilnya sangat bergantung pada sampel negatif mana yang masuk ke test set. CV mean yang dirata-rata dari 100 evaluasi memberikan estimasi yang lebih andal dengan confidence interval 95%.

CV mean F1(0)=0,833 memberikan estimasi yang lebih realistis untuk performa model pada skenario stratified. Namun, perlu ditekankan bahwa estimasi ini **tidak berlaku untuk skenario temporal**. Repeated CV tetap menggunakan stratified sampling yang mencampur angkatan secara acak, sehingga tidak dapat mendeteksi distribution shift antar cohort. CV mean yang tampak baik (F1(0)=0,833) tidak mencerminkan kegagalan total model pada temporal validation (F1(0)=0,000).

### 3.5.4 Fase 4-6: Permutation Importance, Rule Stability, Distribution Shift

**Fase 4 - Permutation importance.** Untuk memvalidasi keandalan feature importance yang diperoleh dari metode MDI (Mean Decrease in Impurity), dilakukan permutation importance dengan 30 kali pengacakan pada stratified test set. Metode ini mengukur penurunan F1-Score ketika nilai suatu fitur diacak secara acak, sehingga memberikan estimasi kontribusi aktual fitur terhadap performa model.

| Rank | Fitur | MDI | Permutation Mean | Permutation Std |
|:----:|-------|:---:|:----------------:|:---------------:|
| 1 | `sks_sem2` | 0,558 | **0,128** | 0,019 |
| 2 | `sks_sem3` | 0,361 | **0,066** | 0,012 |
| 3-14 | 12 fitur lain | 0,047 | **0,000** | 0,000 |

Permutation importance mengkonfirmasi bahwa `sks_sem2` dan `sks_sem3` adalah dua fitur dominan di kedua metode, dengan korelasi peringkat Spearman sebesar 0,675 (p=0,008). Hanya dua fitur yang memberikan kontribusi terukur terhadap F1-Score ketika diacak - `sks_sem2` (penurunan F1 rata-rata 0,128) dan `sks_sem3` (penurunan 0,066). Sebanyak 12 fitur lainnya memiliki permutation importance 0,000, mengindikasikan bahwa model depth=3 yang sangat sederhana hanya mengandalkan dua fitur SKS untuk seluruh keputusan klasifikasi, termasuk `ips_std` dan `avg_ips` yang memiliki MDI non-nol tetapi tidak memberikan kontribusi aktual yang terukur saat diacak. Tidak ada fitur dengan hidden signal yang terlewat oleh MDI - seluruh fitur yang zero-MDI juga zero-permutation.

**Fase 5 - Rule stability.** Stabilitas aturan keputusan diuji dengan melatih model pada 10 fold stratified CV secara terpisah dan mengekstrak decision rules dari setiap fold:

| Metrik Stabilitas | Hasil |
|-------------------|-------|
| Root split `sks_sem2` | **10/10 folds (100%)** |
| Jumlah daun | 7,0 (std=0, range 7-7) |
| Daun prediksi negatif | 1,5 (range 1-2) |
| Fitur selalu digunakan | `ips_sem1`, `sks_sem2`, `sks_sem3`, `failed_courses`, `ips_std` |
| Fitur kadang digunakan | `sks_completion_ratio`, `ips_min` |
| Fitur tidak pernah digunakan | `ips_sem2`, `ips_sem3`, `sks_sem1`, `failed_in_sem1`, `repeated_courses`, `ips_trend`, `avg_ips` |

Root split `sks_sem2` stabil di seluruh 10 fold, menegaskan bahwa fitur ini adalah prediktor paling robust dalam dataset. Jumlah daun konsisten di 7, sesuai dengan constraint `max_depth=3` dan `min_samples_leaf=10`. Lima fitur selalu digunakan di setiap fold, sementara 7 fitur tidak pernah digunakan - konsisten dengan temuan permutation importance bahwa sebagian besar fitur tidak memberikan kontribusi pada model. Namun, stabilitas ini hanya berlaku pada skenario stratified. Model temporal memiliki struktur tree yang sepenuhnya berbeda dengan root split di `failed_courses`, bukan `sks_sem2`.

**Fase 6 - Distribution shift.** Analisis pergeseran distribusi fitur antar angkatan mengungkapkan faktor sistemik yang menyebabkan kegagalan temporal validation:

| Angkatan | sks_sem2 mean | sks_sem3 mean | avg_ips mean | Neg/Total |
|----------|:-------------:|:-------------:|:------------:|:---------:|
| 2015-2019 (train) | 16,0 | 10,7 | 3,14 | 7/291 |
| 2020-2021 (train) | 10,9 | 8,8 | 3,12 | 5/86 |
| **2022** (test) | 17,6 | 9,1 | 2,71 | **5/181** |
| **2023** (test) | 19,2 | 17,8 | 2,93 | **49/50** |

Mean `sks_sem2` pada data latih historis (2015-2021) berkisar antara 10-16, tetapi melonjak menjadi ~19 pada angkatan 2023. Pergeseran ini bersifat fundamental - bukan sekadar variasi acak, melainkan perubahan pola pengambilan SKS antar generasi mahasiswa. Model yang dilatih pada pola SKS rendah (2015-2021) tidak dapat menggeneralisasi ke cohort dengan pola SKS tinggi (2022-2023). Lonjakan drastis pada angkatan 2023 (49/50 negatif, SKS mean 19,2) menciptakan kondisi di mana data test memiliki distribusi yang sangat berbeda dari data train, membuat prediksi berbasis data historis tidak relevan.

Ketiga fase analisis ini bersama-sama mengkonfirmasi kesimpulan utama evaluasi: **model tidak layak untuk deployment.** Permutation importance membuktikan bahwa hanya dua fitur yang benar-benar bekerja. Rule stability membuktikan bahwa aturan hanya stabil dalam skenario stratified. Distribution shift membuktikan bahwa perubahan pola antar cohort membuat model historis tidak relevan. Model stratified memberikan ilusi performa yang baik karena mencampur angkatan secara artifisial, tetapi pada skenario deployment nyata, model gagal total.

## 3.6 Arsitektur Pipeline

Keseluruhan proses dalam penelitian ini mengikuti alur CRISP-DM yang terdiri dari enam fase utama. Setiap fase menghasilkan artefak yang menjadi input bagi fase berikutnya, dengan mekanisme iteratif yang memungkinkan perbaikan berdasarkan temuan dari fase selanjutnya. Arsitektur pipeline dari awal hingga akhir digambarkan sebagai berikut:

```
                        ARSITEKTUR PIPELINE CRISP-DM
    
    Fase 1: Business Understanding
    ===============================
    [Proposal Data Mining] --> Identifikasi tujuan bisnis
                                --> Transformasi ke masalah data mining
                                --> Penetapan metrik evaluasi (Recall(0) >= 0,70,
                                    F1(0) >= 0,50, AUC >= 0,75)
    
    Fase 2: Data Understanding
    ==========================
    [SQL Server 2022 (Docker)] --> 01-schema-discovery.sql
                                    --> Profiling 405 tabel di database LITIGASI
                                    --> Identifikasi 4 tabel relevan (tblMHS, IPSIPK,
                                        Qnilai_mhs, Kul_Kehadiran)
                                    --> Penemuan masalah kualitas data
                                        (placeholder IPS=0.0, bug SKS kumulatif,
                                        NIM mismatch, missing values)
                                    --> Output: exploration-log.md, table mapping
    
    Fase 3: Data Preparation
    ========================
    [extract_dataset.py (424 baris)] --> Bulk query dari SQL Server
                                         --> Join & agregasi di memori (Python)
                                         --> Filtering 1.621 -> 608 mahasiswa
                                         --> Perbaikan bug SKS kumulatif
                                         --> Transformasi fitur turunan (5 derived)
                                         --> Output: dataset.csv (608 x 27)
    
    [Data Cleansing]
    --> Koreksi zero-IPS placeholder (0.0 -> NaN, 36% dataset)
    --> Imputasi global median (bukan per-angkatan)
    --> Drop fitur leakage & non-prediktif (11 kolom)
    --> Output: dataset_clean.csv (608 x 17, 0 NULLs)
    
    [Train-Test Split]
    --> Temporal: train <=2021 (377), test >2021 (231)
    --> Stratified: 80/20 random (486 train, 122 test)
    --> Output: dataset_train.csv, dataset_test.csv
    
    Fase 4: Modeling
    ================
    [Iterasi 1 - Baseline]
    --> Temporal baseline: Recall(0)=0,037 (gagal)
    --> Stratified baseline: Recall(0)=0,929 (overfit, depth=8)
    --> Investigasi SKS: ditemukan temporal proxy dari imputasi per-angkatan
    
    [Iterasi 2 - Global Median]
    --> Perbaikan imputasi: per-angkatan -> global median
    --> Stratified: F1(0)=0,867, Precision(0)=0,813 (+0,163)
    --> Temporal: tetap gagal (F1(0)=0,060)
    
    [Iterasi 3 - Hyperparameter Tuning]
    --> GridSearchCV: 240 kombinasi x 5-fold = 1.200 fit
    --> Best: max_depth=3, min_samples_leaf=10, criterion='gini'
    --> F1(0)=0,889, 7 leaves, 13 nodes, 6 fitur digunakan
    --> Feature engineering (gagal): 4 binary flags -> importance=0
    --> Combined tuning: F1(0) stagnan -> ceiling tercapai
    
    Fase 5: Evaluation
    ==================
    [Fase 1] Temporal Validation -> F1(0)=0,000 (temuan kritis)
    [Fase 2] Deep Error Analysis -> 54/54 FN, 96,3% dari IH, 90,7% angkatan 2023
    [Fase 3] Repeated 10x10-Fold CV -> F1(0) mean=0,833 [CI: 0,814-0,851]
    [Fase 4] Permutation Importance -> sks_sem2 & sks_sem3 dominan (Spearman 0,675)
    [Fase 5] Rule Stability -> root split sks_sem2 di 10/10 folds (100%)
    [Fase 6] Distribution Shift -> sks_sem2 mean ~11 (train) -> ~19 (test 2023)
    
    Fase 6: Deployment Decision
    ============================
    [HASIL EVALUASI] -> Model gagal total pada temporal (F1(0)=0,000)
                         -> 3 faktor: sample imbalance (14 neg), distribution shift,
                            constraint hyperparameter (min_samples_leaf=10)
                         -> **KEPUTUSAN: TIDAK LAYAK DEPLOYMENT**
                         -> Rekomendasi: SMOTE, ensemble methods, periodic retraining
```

Pipeline di atas mengilustrasikan alur kerja dari hulu ke hilir, mulai dari koneksi ke basis data SQL Server 2022 yang berjalan di atas container Docker dengan image `mcr.microsoft.com/mssql/server:2022-latest`, hingga keputusan akhir bahwa model belum siap untuk deployment. Setiap fase memiliki artefak keluaran yang terdokumentasi di folder masing-masing: proposal bisnis di `1-business-understanding/`, hasil eksplorasi dan profiling di `2-data-understanding/`, dataset dan kode preprocessing di `3-data-preparation/`, notebook pemodelan di `4-modeling/`, serta laporan evaluasi lengkap di `5-evaluation/`.

Mekanisme iteratif CRISP-DM tercermin dalam perjalanan antar fase pada proyek ini. Temuan pada fase Modeling (Iterasi 1) bahwa imputasi per-angkatan menciptakan temporal proxy memicu kembali ke fase Data Preparation untuk mengganti strategi imputasi menjadi global median. Demikian pula, temuan pada fase Evaluation bahwa model gagal total pada temporal validation menjadi feedback bagi iterasi modeling lanjutan yang direkomendasikan (SMOTE, ensemble, periodic retraining). Siklus iteratif ini adalah fitur utama CRISP-DM yang membedakannya dari pendekatan linear seperti SEMMA atau KDD.

Komponen teknis utama pipeline meliputi: (1) **ekstraksi data** menggunakan Python dengan koneksi `pymssql` ke SQL Server yang membaca keempat tabel secara massal dan memprosesnya di memori; (2) **preprocessing** yang mencakup koreksi placeholder sistem, imputasi global median, dan eliminasi fitur leakage/non-prediktif; (3) **pemodelan** dengan Decision Tree Classifier dari scikit-learn melalui tiga iterasi bertahap; serta (4) **evaluasi** komprehensif enam fase yang mencakup temporal validation, error analysis, repeated cross-validation, permutation importance, rule stability, dan distribution shift.

Seluruh proses ini terintegrasi dalam pipeline yang reprodusibel: menjalankan `extract_dataset.py` pada folder `3-data-preparation/` akan menghasilkan dataset mentah, yang kemudian diproses melalui notebook-notebook di `3-data-preparation/` dan `4-modeling/` untuk menghasilkan model dan seluruh metrik evaluasi yang dilaporkan dalam penelitian ini.

---

# Bab IV - Hasil dan Pembahasan

## 4.1 Analisis Data Eksploratif (EDA)

Analisis data eksploratif dilakukan untuk memahami karakteristik dataset sebelum memasuki tahap pemodelan. Seluruh visualisasi dan statistik pada bagian ini dihasilkan dari dataset final yang telah melalui proses filtering dan preprocessing, yaitu 608 mahasiswa dengan 14 fitur prediktor dan 1 variabel target. Eksplorasi mencakup analisis distribusi target, distribusi fitur per kelas, serta korelasi antar fitur untuk mengidentifikasi pola awal dan potensi leakage.

### 4.1.1 Distribusi Target dan Class Imbalance

Distribusi kelas target merupakan temuan pertama yang paling mendasar sekaligus paling kritis dalam analisis ini. Dataset final terdiri dari 608 mahasiswa dengan komposisi sebagai berikut:

| Kelas Target | Jumlah | Persentase |
|--------------|-------:|-----------:|
| Tepat Waktu (1) | 540 | 88,8% |
| Tidak Tepat Waktu (0) | 68 | 11,2% |
| **Total** | **608** | **100%** |

Distribusi ini menunjukkan ketidakseimbangan kelas (class imbalance) yang signifikan dengan rasio kelas mayoritas terhadap minoritas sekitar 8:1. Dari seluruh mahasiswa, hanya 11,2% yang tergolong tidak lulus tepat waktu, sementara 88,8% sisanya lulus tepat waktu ([Gambar 2]).

![Distribusi Target Kelas](3-data-preparation/eda-charts/02_target_distribution.png)

**Gambar 2.** Distribusi target kelas menunjukkan dominasi kelas "Tepat Waktu" (88,8%) dibandingkan "Tidak Tepat Waktu" (11,2%).

Tingkat ketidakseimbangan ini membawa implikasi langsung terhadap pemilihan metrik evaluasi model. Pada dataset dengan distribusi seperti ini, akurasi (accuracy) bukanlah metrik yang informatif. Sebuah model yang memprediksi seluruh mahasiswa sebagai "Tepat Waktu" akan mencapai akurasi 88,8% tanpa pernah mendeteksi satu pun mahasiswa berisiko. Metrik tersebut memberikan ilusi performa yang baik namun sama sekali tidak berguna untuk tujuan deteksi dini.

Oleh karena itu, metrik utama yang digunakan dalam penelitian ini difokuskan pada kelas minoritas (Tidak Tepat Waktu), yaitu:

1. **Recall(0)** — Proporsi mahasiswa berisiko yang berhasil terdeteksi oleh model. Metrik ini menjawab pertanyaan paling penting bagi institusi: dari semua mahasiswa yang benar-benar tidak lulus tepat waktu, berapa persen yang berhasil diidentifikasi oleh sistem peringatan dini?

2. **F1(0)** — Harmonic mean antara precision dan recall untuk kelas minoritas. Metrik ini memberikan keseimbangan antara dua aspek: kemampuan mendeteksi mahasiswa berisiko (recall) dan ketepatan deteksi (precision). F1(0) digunakan sebagai primary scoring metric dalam hyperparameter tuning dengan GridSearchCV.

3. **Precision(0)** — Proporsi mahasiswa yang diprediksi berisiko dan benar-benar berisiko. Meskipun penting untuk menghindari false positive yang berlebihan (sumber daya intervensi terbatas), precision tidak boleh dikejar dengan mengorbankan recall.

Ketidakseimbangan kelas pada dataset ini termasuk dalam kategori moderate (rasio 8:1), artinya teknik dasar seperti class weighting atau threshold adjustment mungkin mencukupi tanpa perlu resampling ekstrem. Namun, tantangan utama bukan hanya ketidakseimbangan secara global, melainkan juga ketidakseimbangan yang lebih parah pada skenario temporal split di mana data latih historis hanya memiliki 14 sampel negatif (3,7%) — faktor yang akan dibahas lebih lanjut pada bagian evaluasi.

Distribusi kelas target per program studi juga menunjukkan pola yang berbeda:

| Program | Tepat Waktu | Tidak Tepat | % Negatif |
|---------|------------:|------------:|----------:|
| AP (D3) | 137 | 10 | 6,8% |
| IH (S1) | 403 | 58 | 12,6% |

Mahasiswa program sarjana (IH) memiliki proporsi ketidaktepatan lulus hampir dua kali lipat dibandingkan program vokasi (AP), yaitu 12,6% berbanding 6,8%. Perbedaan ini mengindikasikan bahwa karakteristik akademik antar program studi perlu dipertimbangkan dalam analisis lebih lanjut, meskipun fitur `program` sendiri memiliki korelasi yang rendah terhadap target.

### 4.1.2 Distribusi Fitur per Target

Analisis distribusi fitur per kelas target bertujuan mengidentifikasi pola awal yang membedakan mahasiswa tepat waktu dan tidak tepat waktu berdasarkan performa akademik tiga semester pertama. Fokus utama diberikan pada dua kelompok fitur: Indeks Prestasi Semester (IPS) dan Satuan Kredit Semester (SKS), karena keduanya merupakan prediktor utama yang diperoleh langsung dari tabel `IPSIPK`.

**Distribusi IPS per Semester.** Gambar 4 dan 6 menyajikan distribusi IPS untuk setiap semester, baik secara keseluruhan maupun dipecah per kelas target.

![Distribusi IPS per Semester](3-data-preparation/eda-charts/04_ips_distribution.png)

**Gambar 4.** Distribusi IPS per semester untuk seluruh mahasiswa.

![Distribusi IPS per Kelas Target](3-data-preparation/eda-charts/06_ips_by_target.png)

**Gambar 6.** Distribusi IPS per semester yang dibedakan berdasarkan kelas target (Tepat Waktu vs Tidak Tepat).

Pada Gambar 6 terlihat bahwa mahasiswa tidak tepat waktu cenderung memiliki IPS yang lebih rendah di setiap semester dibandingkan mahasiswa tepat waktu, meskipun perbedaannya tidak terlalu mencolok. Median IPS mahasiswa tepat waktu berkisar di atas 3,0 untuk seluruh tiga semester, sementara median IPS mahasiswa tidak tepat waktu berada sedikit di bawah 3,0 pada semester 1 dan 2, dengan penurunan yang lebih terlihat pada semester 3. Pola ini mengindikasikan bahwa performa akademik (IPS) memang berkontribusi terhadap ketepatan lulus, namun bukan merupakan faktor dominan — indikasi awal bahwa beban studi (SKS) mungkin memainkan peran yang lebih besar.

**Distribusi SKS per Semester.** Gambar 7 menyajikan distribusi SKS yang diambil mahasiswa per semester, dipecah berdasarkan kelas target.

![Distribusi SKS per Semester](3-data-preparation/eda-charts/07_sks_distribution.png)

**Gambar 7.** Distribusi SKS per semester yang dibedakan berdasarkan kelas target.

Gambar 7 mengungkapkan pola yang jauh lebih kontras dibandingkan distribusi IPS. Pada semester 2, mahasiswa tidak tepat waktu memiliki distribusi SKS yang bergeser secara signifikan ke nilai yang lebih tinggi dibandingkan mahasiswa tepat waktu. Rata-rata SKS semester 2 untuk mahasiswa tidak tepat waktu adalah **19,43**, sedangkan mahasiswa tepat waktu hanya **12,96** — selisih mencapai 6,47 SKS. Korelasi Pearson antara `sks_sem2` dan target mencapai -0,348, menjadikannya fitur dengan korelasi tertinggi terhadap kelas minoritas di antara seluruh prediktor.

Lebih penting lagi, pada semester 3 terlihat pola sebaliknya: mahasiswa tidak tepat waktu cenderung mengalami penurunan jumlah SKS yang drastis. Rata-rata SKS semester 3 untuk kelompok tidak tepat waktu turun dari 19,43 menjadi sekitar 18,15, sementara mahasiswa tepat waktu justru cenderung stabil atau sedikit meningkat dibandingkan semester 2. Pola ini konsisten dengan fenomena yang disebut **"overload lalu collapse"** — mahasiswa mengambil beban SKS yang sangat tinggi pada semester awal (khususnya semester 2), namun tidak mampu mempertahankannya dan mengalami penurunan drastis pada semester berikutnya, yang menjadi indikator awal ketidaktepatan lulus.

Perbandingan numerik fitur SKS dan IPS antar kelas dirangkum dalam tabel berikut:

| Fitur | Tepat Waktu (mean) | Tidak Tepat (mean) | Selisih |
|-------|:------------------:|:------------------:|:-------:|
| sks_sem1 | 14,02 | 15,47 | +1,45 |
| sks_sem2 | 12,96 | **19,43** | **+6,47** |
| sks_sem3 | 12,54 | 18,15 | +5,61 |
| ips_sem1 | 3,12 | 2,89 | -0,23 |
| ips_sem2 | 3,08 | 2,87 | -0,21 |
| ips_sem3 | 3,05 | 2,78 | -0,27 |

Dari tabel di atas, terlihat dengan jelas bahwa pemisah utama antar kelas adalah fitur SKS, bukan IPS. Selisih SKS semester 2 mencapai 6,47 poin — jauh melampaui selisih IPS yang hanya berkisar 0,21–0,27 poin. Bahkan pada semester 1, sebelum pola overload muncul, mahasiswa tidak tepat waktu sudah mengambil SKS lebih banyak (15,47 vs 14,02), mengindikasikan bahwa kecenderungan overload sudah dimulai sejak awal perkuliahan.

Fitur turunan juga menunjukkan pola yang mendukung temuan ini. Mahasiswa tidak tepat waktu memiliki rata-rata `failed_courses` yang lebih tinggi (sekitar 3,09 mata kuliah gagal) dibandingkan mahasiswa tepat waktu, serta volatilitas IPS (`ips_std`) yang lebih besar, mengindikasikan inkonsistensi performa akademik. Temuan awal ini menjadi dasar bagi iterasi pemodelan dan aturan keputusan yang akan dibahas pada bagian selanjutnya.

### 4.1.3 Analisis Korelasi

Analisis korelasi dilakukan untuk mengukur hubungan linear antar fitur (multikolinearitas) serta hubungan setiap fitur terhadap target (ketepatan lulus). Tujuan utama analisis ini adalah mengidentifikasi fitur-fitur yang redundan atau bersifat leakage, serta memvalidasi kelayakan penggunaan Decision Tree yang robust terhadap multikolinearitas.

**Heatmap korelasi antar fitur.** Gambar 11 menyajikan matriks korelasi Pearson untuk seluruh fitur dalam dataset.

![Heatmap Korelasi Antar Fitur](3-data-preparation/eda-charts/11_correlation_heatmap.png)

**Gambar 11.** Heatmap korelasi Pearson antar seluruh fitur dalam dataset. Warna merah menunjukkan korelasi positif, biru menunjukkan korelasi negatif.

Dari heatmap terlihat beberapa kelompok korelasi yang perlu diperhatikan:

**Multikolinearitas antar fitur IPS.** Fitur `ips_sem1`, `ips_sem2`, dan `ips_sem3` menunjukkan korelasi satu sama lain dalam rentang moderat 0,3–0,6. Hubungan ini wajar karena performa akademik seorang mahasiswa cenderung konsisten antar semester — mahasiswa dengan IPS tinggi di semester 1 cenderung mempertahankan performa di semester berikutnya. Namun, tingkat korelasi ini tidak tergolong tinggi (di bawah 0,8) sehingga tidak menimbulkan masalah multikolinearitas serius yang dapat mengganggu stabilitas estimasi koefisien. Gambar 13 menyajikan visualisasi khusus untuk interkorelasi IPS antar semester.

![Interkorelasi IPS Antar Semester](3-data-preparation/eda-charts/13_ips_intercorrelation.png)

**Gambar 13.** Interkorelasi antar fitur IPS tiga semester pertama.

**Korelasi fitur dengan target.** Gambar 12 menyajikan korelasi setiap fitur terhadap variabel target `tepat_waktu`.

![Korelasi Fitur dengan Target](3-data-preparation/eda-charts/12_correlation_with_target.png)

**Gambar 12.** Korelasi Pearson setiap fitur terhadap target ketepatan lulus. Garis putus-putus menunjukkan threshold korelasi signifikan.

Temuan paling kritis dari analisis korelasi dengan target adalah:

1. **Leakage `ips_sem4` (r=0,877).** Korelasi `ips_sem4` dengan target mencapai 0,877 — jauh melampaui fitur lainnya. Nilai setinggi ini merupakan indikasi kuat data leakage (kebocoran data), bukan sinyal prediktif yang genuine. Seorang mahasiswa hanya memiliki data IPS semester 4 jika ia masih aktif terdaftar pada semester tersebut. Mahasiswa yang lulus tepat waktu secara otomatis memiliki data semester 4 karena mereka bertahan hingga semester tersebut. Sebaliknya, mahasiswa yang dropout atau lulus terlambat mungkin tidak memiliki data semester 4 yang lengkap. Dengan kata lain, `ips_sem4` tidak memprediksi ketepatan lulus — ia adalah konsekuensi dari ketepatan lulus itu sendiri. Fitur ini bersama dengan `sks_sem4`, `ipk_sem4`, dan `semester_count` dikeluarkan dari dataset sebelum pemodelan.

2. **Korelasi negatif SKS dengan target.** Fitur `sks_sem2` (r=-0,348) dan `sks_sem3` (r=-0,282) memiliki korelasi negatif tertinggi terhadap target, mengonfirmasi temuan dari analisis distribusi bahwa semakin tinggi beban SKS, semakin tinggi risiko ketidaktepatan lulus. Korelasi ini bersifat genuine karena mencerminkan hubungan kausal antara beban studi berlebih dengan kegagalan menyelesaikan studi tepat waktu.

3. **Korelasi rendah fitur demografis.** Fitur seperti `id_agama` (r=0,031), `jenis_kelamin` (r=0,050), dan `avg_attendance` (r=-0,016) memiliki korelasi mendekati nol terhadap target. Fitur-fitur ini tidak memberikan kontribusi informatif untuk membedakan kelas dan dikeluarkan dari dataset pada tahap data cleansing.

Tabel berikut merangkum korelasi seluruh fitur terhadap target:

| Fitur | Korelasi dengan Target | Kategori |
|-------|:---------------------:|:--------:|
| ips_sem4 | **+0,877** ⚠️ | Leakage — dihapus |
| sks_sem2 | **-0,348** | Prediktor utama |
| sks_sem3 | -0,282 | Prediktor utama |
| ips_sem4 dihapus | - | - |
| ips_sem1 | -0,270 | Prediktor sekunder |
| ips_sem3 | -0,240 | Prediktor sekunder |
| ips_sem2 | -0,200 | Prediktor sekunder |
| sks_sem1 | -0,120 | Signal lemah |
| id_agama | +0,031 | Non-prediktif — dihapus |
| jenis_kelamin | +0,050 | Non-prediktif — dihapus |
| avg_attendance | -0,016 | Non-prediktif — dihapus |

**Justifikasi penggunaan Decision Tree.** Temuan adanya multikolinearitas moderat antar fitur IPS (r=0,3–0,6) bukanlah masalah bagi algoritma Decision Tree yang digunakan dalam penelitian ini. Berbeda dengan model linear seperti regresi logistik yang mengasumsikan independensi antar fitur, Decision Tree bekerja dengan memilih satu fitur terbaik pada setiap split secara sekuensial. Jika dua fitur berkorelasi tinggi, tree cukup memilih salah satu yang paling informatif pada level root, dan fitur lainnya mungkin tidak akan pernah digunakan. Hal ini terlihat pada model final di mana hanya 6 dari 14 fitur yang benar-benar digunakan oleh tree, dan beberapa fitur IPS yang berkorelasi tidak dipilih karena informasi yang dibawanya sudah tercakup oleh fitur lain yang lebih dominan. Dengan demikian, multikolinearitas tidak mengganggu proses pemodelan maupun validitas aturan keputusan yang dihasilkan.

Korelasi `ips_sem4` dengan target sebesar 0,877 menjadi temuan penting yang mengonfirmasi bahwa data semester 4 dan seterusnya tidak boleh digunakan sebagai prediktor. Keputusan untuk membatasi fitur pada tiga semester pertama adalah langkah yang tepat untuk menjaga integritas model dan menghindari leakage yang dapat menyebabkan overestimasi performa secara artifisial.

## 4.2 Iterasi 1: Baseline Modeling

Iterasi pertama pemodelan bertujuan membangun Decision Tree awal dengan konfigurasi default scikit-learn untuk memperoleh gambaran dasar mengenai kemampuan algoritma dalam mengklasifikasikan ketepatan lulus. Tiga eksperimen dilakukan secara paralel pada iterasi ini: temporal baseline, stratified baseline, dan investigasi SKS. Seluruh eksperimen pada Iterasi 1 menggunakan imputasi per-angkatan (median dihitung terpisah untuk setiap angkatan) dan 16 fitur prediktor (termasuk `angkatan` dan `program`).

### 4.2.1 Temporal Baseline — Kegagalan Awal

Eksperimen pertama menggunakan temporal split sebagai skenario validasi yang paling mendekati kondisi deployment nyata. Model Decision Tree dengan parameter default (`max_depth=None`, `min_samples_leaf=1`, `criterion='gini'`) dilatih pada data historis angkatan <=2021 dan diuji pada angkatan >2021 yang belum pernah dilihat model.

**Komposisi data latih.** Temporal train terdiri dari 377 mahasiswa dengan ketidakseimbangan kelas yang sangat ekstrem: hanya **14 sampel negatif** (3,7%) dari total 377 mahasiswa. Sebanyak 363 mahasiswa (96,3%) adalah kelas positif (Tepat Waktu). Ketidakseimbangan ini jauh lebih parah dibandingkan ketidakseimbangan global dataset (11,2%), karena sebagian besar mahasiswa tidak tepat waktu justru berasal dari angkatan 2022-2023 yang masuk ke dalam data uji.

**Hasil pengujian.** Performa model pada temporal test (231 mahasiswa, 54 negatif) menunjukkan kegagalan yang nyaris total:

| Metrik | Nilai |
|--------|:-----:|
| Accuracy | 0,7749 |
| Precision(0) | 1,0000 |
| Recall(0) | **0,0370** |
| F1(0) | **0,0714** |
| AUC | **0,5185** |

Confusion matrix model temporal baseline disajikan pada Gambar 17.

![Confusion Matrix Baseline Temporal](4-modeling/1-baseline/charts/01-confusion-matrix.png)

**Gambar 17.** Confusion matrix model Decision Tree baseline pada temporal split. Model hanya berhasil mendeteksi 2 dari 54 mahasiswa berisiko.

Dari 54 mahasiswa tidak tepat waktu, hanya **2 mahasiswa (3,7%)** yang berhasil terdeteksi (True Negative), sementara **52 mahasiswa (96,3%)** terlewatkan sebagai False Negative. Model tidak menghasilkan satu pun False Positive (Precision=1,000), yang menunjukkan bahwa model sangat konservatif — hanya membuat prediksi negatif ketika benar-benar yakin, dan sisanya memprediksi semua sebagai "Tepat Waktu". Akurasi 77,49% tercapai semata-mata karena model benar mengklasifikasikan 177 mahasiswa tepat waktu.

**Feature importance.** Feature importance model temporal baseline disajikan pada Gambar 18.

![Feature Importance Baseline Temporal](4-modeling/1-baseline/charts/01-feature-importance.png)

**Gambar 18.** Feature importance model Decision Tree baseline pada temporal split. Fitur `ips_min` mendominasi dengan kontribusi tertinggi.

Distribusi feature importance menunjukkan bahwa model mengandalkan fitur yang berbeda dari ekspektasi awal. Fitur `ips_min` (IPS terendah) menjadi fitur paling dominan, bukan fitur SKS seperti yang diperkirakan dari analisis EDA. Hal ini terjadi karena terbatasnya sampel minoritas menyebabkan tree membuat split berdasarkan fitur yang secara kebetulan dapat memisahkan beberapa sampel negatif di data latih, tanpa benar-benar menangkap pola yang generalizable. Lima dari 16 fitur memiliki kontribusi nol, mengindikasikan bahwa tree tidak memiliki cukup data untuk memanfaatkan seluruh informasi yang tersedia.

**Struktur pohon.** Struktur Decision Tree yang dihasilkan disajikan pada Gambar 19.

![Struktur Decision Tree Baseline Temporal](4-modeling/1-baseline/charts/01-decision-tree.png)

**Gambar 19.** Struktur Decision Tree baseline pada temporal split dengan kedalaman 9 dan 21 daun.

Pohon yang dihasilkan memiliki kompleksitas tinggi: kedalaman **9 level**, **21 daun**, dan **41 node**, dengan 11 dari 16 fitur yang digunakan. Hanya sebagian kecil dari data latih (sekitar 1,6%) yang mengalir ke cabang dengan prediksi kelas minoritas, sementara sisanya berakhir di daun mayoritas. Kompleksitas ini merupakan bentuk overfitting — pohon menggali terlalu dalam untuk menemukan pola pada 14 sampel negatif, menghasilkan struktur yang terlalu spesifik terhadap data latih dan tidak dapat menggeneralisasi ke data uji.

**Analisis kegagalan.** Kegagalan temporal baseline disebabkan oleh satu akar masalah fundamental: **jumlah sampel minoritas yang tidak mencukupi**. Dengan hanya 14 sampel negatif (3,7%) di data latih, Decision Tree tidak memiliki cukup representasi kelas minoritas untuk mempelajari pola yang membedakan mahasiswa berisiko dari mahasiswa aman. Decision Tree bekerja dengan membagi data secara rekursif — setiap split membagi data menjadi subset yang lebih kecil. Dengan 14 sampel negatif, setelah beberapa level split, jumlah sampel minoritas di setiap cabang menjadi sangat sedikit sehingga split berikutnya tidak lagi informatif. Akibatnya, tree lebih memilih untuk mengklasifikasikan hampir seluruh sampel sebagai kelas mayoritas, yang menghasilkan recall mendekati nol.

Temuan ini mengonfirmasi bahwa temporal split dengan data historis yang terbatas (hanya 14 negatif dari angkatan <=2021) tidak dapat digunakan secara langsung tanpa penanganan class imbalance. Eksperimen selanjutnya pada Iterasi 1 beralih ke stratified split untuk menguji apakah penambahan sampel minoritas di data latih dapat memperbaiki performa model.

### 4.2.2 Stratified Baseline — Performa Tinggi yang Menyesatkan

Eksperimen kedua mengganti temporal split dengan stratified random split (80/20, stratify=target) untuk mengatasi kelangkaan sampel minoritas di data latih. Variasi ini mempertahankan proporsi kelas yang sama antara training dan testing, sehingga kelas minoritas mendapatkan representasi yang lebih baik. Fitur `angkatan` dan `program` di-drop untuk mencegah data leakage karena angkatan 2023 seluruhnya merupakan kelas negatif.

**Komposisi data.** Dengan stratified split, distribusi kelas menjadi seimbang antara training dan testing:

| Split | Total | Negatif (%) | Positif (%) |
|-------|:----:|:-----------:|:-----------:|
| Train | 486 | 54 (11,1%) | 432 (88,9%) |
| Test | 122 | 14 (11,5%) | 108 (88,5%) |

Penambahan sampel negatif di training dari **14 menjadi 54** merupakan peningkatan signifikan yang berpotensi memberikan Decision Tree lebih banyak data untuk mempelajari pola kelas minoritas. Namun, perlu dicatat bahwa distribusi ini diperoleh dengan mengacak waktu, bukan mensimulasikan prediksi ke masa depan.

**Hasil pengujian.** Performa model pada stratified test menunjukkan peningkatan dramatis dari temporal baseline:

| Metrik | Temporal Baseline | Stratified Baseline | Delta |
|--------|:-----------------:|:-------------------:|:-----:|
| Accuracy | 0,7749 | **0,9344** | +0,1595 |
| Precision(0) | 1,0000 | **0,6500** | -0,3500 |
| Recall(0) | 0,0370 | **0,9286** | +0,8916 |
| F1(0) | 0,0714 | **0,7647** | +0,6933 |
| AUC | 0,5185 | **0,9319** | +0,4134 |

Recall kelas 0 melonjak dari **3,7% menjadi 92,9%** — peningkatan yang tampak luar biasa. Confusion matrix stratified baseline disajikan pada Gambar 23.

![Confusion Matrix Stratified Baseline](4-modeling/1-baseline/02-stratified-results_files/02-stratified-results_5_0.png)

**Gambar 23.** Confusion matrix model Decision Tree pada stratified split. Dari 14 mahasiswa tidak tepat waktu, 13 berhasil terdeteksi (True Negative), namun 7 mahasiswa tepat waktu salah diklasifikasikan sebagai berisiko (False Positive).

Dari 14 mahasiswa tidak tepat waktu, model berhasil mendeteksi **13 mahasiswa (92,9%)** — peningkatan drastis dari hanya 2 mahasiswa pada temporal baseline. Namun, Precision(0) turun menjadi 0,650, artinya **35% prediksi kelas minoritas adalah False Positive**. Model sekarang cenderung terlalu agresif dalam memprediksi risiko, kebalikan dari temporal baseline yang terlalu konservatif.

**Indikasi overfitting.** Kinerja sempurna pada data latih menjadi tanda bahaya pertama:

| Set | Accuracy | Precision(0) | Recall(0) | F1(0) | AUC |
|:---:|:--------:|:------------:|:---------:|:-----:|:---:|
| Train | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **1,0000** |
| Test | 0,9344 | 0,6500 | 0,9286 | 0,7647 | 0,9319 |

Model mencapai performa sempurna pada training (Acc=1,0, seluruh metrik=1,0) — indikasi bahwa Decision Tree menghafal data latih tanpa generalisasi yang memadai. Validasi 10-fold cross-validation mengonfirmasi adanya gap yang konsisten:

| Metrik | Train (CV) | Test (CV) | Gap |
|--------|:----------:|:---------:|:---:|
| Accuracy | 1,0000 | 0,9588 | +0,0412 |
| Recall | 1,0000 | 0,9745 | +0,0255 |
| F1 | 1,0000 | 0,9767 | +0,0233 |
| AUC | 1,0000 | 0,9006 | **+0,0994** |

Gap AUC sebesar ~0,10 menunjukkan bahwa model kehilangan kemampuan diskriminatif yang cukup besar saat dihadapkan pada data baru, meskipun metrik lainnya tampak stabil.

**Struktur pohon.** Decision Tree yang dihasilkan memiliki kedalaman **8 level**, **23 daun**, dan **45 node**, dengan 12 dari 14 fitur yang digunakan. Feature importance model stratified baseline disajikan pada Gambar 24.

![Feature Importance Stratified Baseline](4-modeling/1-baseline/02-stratified-results_files/02-stratified-results_7_1.png)

**Gambar 24.** Feature importance model Decision Tree pada stratified split. Fitur `sks_sem2` mendominasi dengan kontribusi 62,1%.

Distribusi feature importance mengungkapkan pola yang sangat berbeda dari temporal baseline. Fitur `sks_sem2` mendominasi dengan kontribusi **62,1%** — sebuah ketimpangan yang mencolok. Dua fitur (`failed_in_sem1`, `sks_completion_ratio`) memiliki kontribusi nol dan diabaikan sepenuhnya oleh tree. Dominasi `sks_sem2` ini menjadi petunjuk kunci untuk mengungkap akar masalah.

**Root cause: Temporal proxy melalui imputasi per-angkatan.** Dominasi `sks_sem2` (62,1%) dan struktur pohon yang disajikan pada Gambar 25 mengungkap sebuah masalah fundamental.

![Decision Tree Stratified Baseline](4-modeling/1-baseline/02-stratified-results_files/02-stratified-results_8_0.png)

**Gambar 25.** Struktur Decision Tree stratified baseline (kedalaman 8) dengan root split pada fitur `sks_sem2 <= 18,5`.

Root split `sks_sem2 <= 18,5` menjadi kunci untuk memahami mengapa model tampak berkinerja tinggi. Pada data yang diimputasi per-angkatan, nilai SKS semester 2 memiliki korelasi dengan angkatan: angkatan yang lebih baru cenderung mengambil SKS lebih tinggi sebagai akibat dari perubahan kebijakan kurikulum. Karena stratified split mengacak angkatan secara random, pola temporal ini muncul sebagai sinyal prediktif yang kuat — padahal sinyal tersebut bukan karakteristik individu mahasiswa, melainkan artefak dari perbedaan antar angkatan yang sebenarnya bersifat temporal.

Dengan kata lain, stratified split memberikan keuntungan yang tidak realistis kepada model dengan membuat pola temporal (perbedaan angkatan) tampak sebagai pola yang bisa dipelajari. Dalam skenario deployment nyata di mana model harus memprediksi mahasiswa angkatan mendatang, pola temporal dari angkatan sebelumnya belum tentu berlaku.

**Kesimpulan.** Peningkatan performa stratified baseline bersifat menyesatkan karena dua alasan:

1. **Overfitting masif.** Model menghafal data latih (Acc=1,0, depth=8) dan kehilangan kemampuan generalisasi yang terlihat dari gap AUC ~0,10 pada cross-validation.
2. **Temporal proxy.** Dominasi `sks_sem2` (62,1%) sebagai fitur root split mengindikasikan bahwa model memanfaatkan pola antar-angkatan yang sebenarnya bersifat temporal — bukan sinyal prediktif genuine dari karakteristik individu mahasiswa. Imputasi per-angkatan menyebabkan nilai SKS terikat dengan angkatan, dan stratified split yang mengacak waktu membuat pola ini terlihat sebagai korelasi yang valid.

Eksperimen stratified baseline menegaskan bahwa peningkatan jumlah sampel minoritas saja tidak cukup tanpa mempertimbangkan aspek temporal data. Diperlukan iterasi lanjutan untuk menguji apakah penggunaan imputasi global (tanpa ketergantungan angkatan) dapat menghilangkan temporal proxy dan memberikan estimasi performa yang lebih realistis.

## 4.3 Iterasi 2: Global Median Imputation

Iterasi 2 menguji hipotesis yang ditemukan pada akhir Iterasi 1: apakah dominasi `sks_sem2` (62,1%) pada stratified baseline disebabkan oleh imputasi per-angkatan yang menciptakan temporal proxy? Strategi imputasi diubah dari median per-angkatan menjadi **global median** — satu nilai median tunggal untuk seluruh dataset tanpa memandang angkatan. Dengan demikian, mahasiswa dari angkatan mana pun yang memiliki missing values mendapat nilai imputasi yang sama, sehingga tidak ada informasi temporal yang bocor ke dalam fitur.

Pada **temporal split**, hasil pengujian menunjukkan perbaikan yang sangat marginal:

| Metrik | Iterasi 1 | Iterasi 2 | Delta |
|--------|:---------:|:---------:|:-----:|
| Recall(0) | 0,0370 | **0,0556** | +0,0186 |
| F1(0) | 0,0714 | **0,1053** | +0,0339 |
| AUC | 0,5185 | **0,5278** | +0,0093 |

Jumlah mahasiswa berisiko yang terdeteksi meningkat dari 2 menjadi 3 dari total 54 (Gambar 27). Struktur pohon (depth=9, leaves=21) dan dominasi `ips_min` (24,7%) tidak berubah — masalah fundamental tetap pada kelangkaan sampel minoritas (14 negatif), bukan strategi imputasi.

![Confusion Matrix Temporal Global Median](4-modeling/2-global-median/01-temporal-results_files/01-temporal-results_7_0.png)

**Gambar 27.** Confusion matrix model Decision Tree temporal split dengan global median imputation. Model mendeteksi 3 dari 54 mahasiswa berisiko — peningkatan 1 deteksi dari Iterasi 1.

Pada **stratified split**, perubahan imputasi memberikan dampak yang lebih nyata — khususnya pada presisi:

| Metrik | Iterasi 1 | Iterasi 2 | Delta |
|--------|:---------:|:---------:|:-----:|
| Precision(0) | 0,6500 | **0,8125** | +0,1625 |
| Recall(0) | 0,9286 | 0,9286 | ±0,0000 |
| F1(0) | 0,7647 | **0,8667** | +0,1020 |

Recall(0) tetap stabil di 92,9% (13/14 terdeteksi), namun False Positive berkurang dari 7 menjadi 3 (Gambar 30). Distribusi feature importance menjadi lebih seimbang (Gambar 31): `sks_sem2` turun dari 62,1% menjadi 43,2% dan `sks_sem3` naik dari 9,5% menjadi 28,0%, menandakan bahwa temporal proxy melalui imputasi per-angkatan telah berkurang signifikan.

![Confusion Matrix Stratified Global Median](4-modeling/2-global-median/02-stratified-results_files/02-stratified-results_5_0.png)

**Gambar 30.** Confusion matrix stratified global median. False Positive berkurang dari 7 menjadi 3 dibandingkan Iterasi 1.

![Feature Importance Stratified Global Median](4-modeling/2-global-median/02-stratified-results_files/02-stratified-results_7_1.png)

**Gambar 31.** Feature importance stratified global median. `sks_sem2` menurun dari 62,1% menjadi 43,2%.

**Konfirmasi.** Global median imputation memperbaiki presisi stratified split dengan menghilangkan temporal proxy, namun tidak memperbaiki temporal split. Temporal Recall(0) hanya 0,056 berbanding stratified 0,929 — kesenjangan F1(0) sebesar 0,762 poin. Hal ini mengonfirmasi bahwa akar masalah bukan pada strategi imputasi, melainkan pada **fundamental split strategy**: stratified split menyediakan 54 sampel negatif di training (realistis untuk eksperimen tapi tidak untuk deployment), sementara temporal split hanya memiliki 14 sampel negatif. Dengan demikian, seluruh iterasi selanjutnya menggunakan global median sebagai standar imputasi, dengan temporal split sebagai gold standard validasi.

## 4.4 Iterasi 3: Hyperparameter Tuning

Iterasi 3 bertujuan menemukan konfigurasi hyperparameter Decision Tree yang optimal untuk dataset dengan ketidakseimbangan kelas tinggi. Tiga pendekatan tuning dilakukan secara berurutan: pre-pruning tuning menggunakan GridSearchCV, eksperimen feature engineering, dan combined tuning untuk mencari batas performa maksimum.

### 4.4.1 Pre-Pruning Tuning

GridSearchCV dilakukan dengan 5-fold stratified cross-validation menggunakan F1(0) sebagai scoring metric. Grid hyperparameter mencakup 240 kombinasi: `max_depth=[3,4,5,6,None]`, `min_samples_leaf=[5,10,15,20]`, `min_samples_split=[5,10,20]`, `class_weight=[None,'balanced']`, dan `criterion=['gini','entropy']` — total 1.200 fit.

**Best hyperparameters.** Konfigurasi terbaik yang ditemukan:

| Hyperparameter | Nilai |
|----------------|:-----:|
| max_depth | **3** |
| min_samples_leaf | **10** |
| min_samples_split | 5 |
| criterion | **gini** |
| class_weight | None |

**Analisis post-pruning.** Setelah GridSearchCV, post-pruning dengan `ccp_alpha` diuji untuk menyederhanakan tree lebih lanjut. Dari 7 kandidat alpha (rentang 0,000–0,055), alpha optimal adalah 0,000 — artinya pre-pruning yang dihasilkan GridSearchCV sudah optimal dan tidak perlu pemangkasan tambahan (Gambar 33).

![Post-Pruning ccp_alpha vs CV F1(0)](4-modeling/3-hyperparameter-tuning/01-tuning-results_files/01-tuning-results_9_0.png)

**Gambar 33.** Plot post-pruning `ccp_alpha` terhadap CV F1(0). Alpha optimal = 0,000 — pruning tidak memberikan perbaikan.

**Hasil pengujian.** Model terbaik diuji pada stratified test set:

| Metrik | Train | Test |
|--------|:-----:|:----:|
| Accuracy | 0,9691 | **0,9754** |
| Precision(0) | 0,8980 | **0,9231** |
| Recall(0) | 0,8148 | **0,8571** |
| F1(0) | 0,8544 | **0,8889** |
| AUC | 0,9016 | **0,9239** |

Model tuned menunjukkan perilaku yang sehat: semua metrik test set lebih baik dari train set, mengindikasikan **tidak terjadi overfitting**. Confusion matrix test set disajikan pada Gambar 34: 12 dari 14 mahasiswa berisiko terdeteksi (Recall=85,7%), dengan hanya 1 False Positive dari 108 mahasiswa tepat waktu (Precision=92,3%).

![Confusion Matrix Model Tuned](4-modeling/3-hyperparameter-tuning/01-tuning-results_files/01-tuning-results_14_0.png)

**Gambar 34.** Confusion matrix model Decision Tree tuned (pre-pruning) pada stratified test. Model mendeteksi 12/14 mahasiswa berisiko dengan hanya 1 False Positive.

**Feature importance.** Distribusi feature importance model tuned disajikan pada Gambar 35.

![Feature Importance Model Tuned](4-modeling/3-hyperparameter-tuning/01-tuning-results_files/01-tuning-results_18_1.png)

**Gambar 35.** Feature importance model Decision Tree hasil tuning. Dua fitur SKS mendominasi 91,9% dari total importance.

Dibandingkan dengan baseline (Iterasi 2 Global Median Stratified), distribusi feature importance menjadi lebih ekstrem:

| Fitur | Baseline (Iterasi 2) | Tuned (Iterasi 3) | Delta |
|-------|:-------------------:|:-----------------:|:-----:|
| sks_sem2 | 0,432 | **0,558** | +0,126 |
| sks_sem3 | 0,280 | **0,361** | +0,081 |
| ips_std | 0,066 | 0,047 | -0,019 |
| ips_sem1 | 0,079 | 0,004 | -0,075 |
| Lainnya | 0,143 | 0,029 | -0,114 |

Dominasi SKS semakin menguat: `sks_sem2` (55,8%) dan `sks_sem3` (36,1%) kini mencakup **91,9%** dari total importance — meningkat dari 71,2% pada baseline. Hanya 6 dari 14 fitur yang benar-benar digunakan oleh model. Fitur IPS yang sebelumnya memiliki kontribusi (ips_sem1=7,9%, ips_sem2=3,9%) praktis diabaikan setelah tuning. Temuan ini mengonfirmasi bahwa **beban studi (SKS) jauh lebih determinan daripada nilai akademik (IPS)** dalam memprediksi ketepatan lulus.

**Struktur pohon.** Model akhir memiliki struktur yang sangat sederhana dan interpretable: kedalaman 3 level, 7 daun, dan 13 node, menggunakan 6 dari 14 fitur (Gambar 36). Aturan keputusan yang dihasilkan mudah dipahami oleh pihak non-teknis: root split pada `sks_sem2 <= 18,5` — mahasiswa yang mengambil SKS semester 2 lebih dari 18,5 langsung masuk jalur risiko.

![Struktur Decision Tree Model Tuned](4-modeling/3-hyperparameter-tuning/01-tuning-results_files/01-tuning-results_20_0.png)

**Gambar 36.** Struktur Decision Tree model tuned dengan kedalaman 3 dan 7 daun. Setiap daun menampilkan proporsi kelas dan jumlah sampel.

### 4.4.2 Feature Engineering (Gagal)

Setelah pre-pruning tuning menghasilkan model optimal, dilakukan eksperimen penambahan fitur rekayasa (feature engineering) untuk melihat apakah informasi tambahan dapat meningkatkan performa model. Empat fitur baru ditambahkan ke dataset:

- **Binary flags (3 fitur):**
  - `sks_sem2_high` — 1 jika SKS semester 2 di atas rata-rata
  - `sks_sem3_high` — 1 jika SKS semester 3 di atas rata-rata
  - `ips_sem1_low` — 1 jika IPS semester 1 di bawah rata-rata
- **Aggregate feature (1 fitur):**
  - `sks_total` — Total SKS tiga semester (sks_sem1 + sks_sem2 + sks_sem3)

**Hasil.** Seluruh fitur baru memiliki feature importance 0,0000. Decision Tree sama sekali tidak menggunakan binary flags karena algoritma sudah menemukan threshold optimal sendiri pada fitur kontinu yang ada. Misalnya, threshold `sks_sem2 <= 18,5` pada root split sudah secara implisit mengkodekan informasi yang sama dengan binary flag `sks_sem2_high`. Fitur `sks_total` juga redundan karena informasi total SKS sudah tercakup oleh ketiga fitur SKS per semester.

Perbandingan performa sebelum dan sesudah penambahan fitur:

| Metrik | 01-Tuned (14 fitur) | 02-Engineered (18 fitur) | Delta |
|--------|:------------------:|:------------------------:|:-----:|
| F1(0) | 0,8889 | 0,8889 | 0,0000 |
| Recall(0) | 0,8571 | 0,8571 | 0,0000 |
| Precision(0) | 0,9231 | 0,9231 | 0,0000 |
| AUC | 0,9239 | 0,9239 | 0,0000 |
| Train Acc | 0,9691 | 0,9691 | 0,0000 |
| Depth | 3 | 4 | +1 |
| Leaves | 7 | 9 | +2 |
| Features Used | 6/14 | 6/18 | 0 |

Seluruh metrik klasifikasi identik — nilai F1(0), Recall(0), Precision(0), dan AUC tidak berubah sama sekali. Meskipun struktur pohon sedikit berbeda (depth 3→4, leaves 7→9), jumlah fitur yang benar-benar digunakan tetap 6 — hanya fitur asli yang dipilih, tidak ada satu pun fitur baru yang dimanfaatkan.

**Kesimpulan.** Seluruh fitur engineering dihapus dan tidak digunakan pada eksperimen selanjutnya. Hasil ini mengonfirmasi bahwa **14 fitur dasar sudah fully capture seluruh informasi yang diperlukan** untuk memprediksi ketepatan lulus. Penambahan fitur rekayasa tidak memberikan nilai tambah karena informasi yang dikodekan oleh binary flags dan aggregate sudah secara implisit terkandung dalam fitur kontinu asli.

### 4.4.3 Combined Tuning dan Ceiling

Eksperimen ketiga menggabungkan dua hyperparameter yang belum di-tuning sebelumnya: `max_features` dan `min_impurity_decrease`. Tujuan dari eksperimen ini adalah mengatasi dua kelemahan model 01-Tuned: konsentrasi fitur yang ekstrem (top-2=92%) dan potensi cosmetic splits pada kedalaman lebih dalam.

**GridSearchCV.** Grid mencakup 28 kombinasi dengan `max_depth=7` dan `min_samples_leaf=10` (dari hasil pre-pruning):

| Hyperparameter | Nilai yang Diuji |
|----------------|:----------------:|
| max_features | None, 'sqrt', 4, 5 |
| min_impurity_decrease | 0,0; 0,001; 0,002; 0,005; 0,008; 0,01; 0,02 |

Total 140 fit (28 kombinasi × 5-fold CV).

**Insight awal: impurity landscape.** Sebelum GridSearchCV, dilakukan analisis impurity decrease pada tree tidak terkendali (depth=7, tanpa `min_impurity_decrease`). Dari 12 internal nodes, 8 bersifat cosmetic (kedua anak memprediksi kelas yang sama) — artinya 66% split tidak mengubah keputusan. Temuan ini memotivasi penggunaan `min_impurity_decrease` untuk memblokir split noise pada kedalaman tertentu (Gambar 39).

![Impurity Landscape Analysis](4-modeling/3-hyperparameter-tuning/03-combined-results_files/03-combined-results_5_0.png)

**Gambar 39.** Analisis impurity decrease pada tree depth=7. Split dengan impurity decrease rendah (merah) bersifat cosmetic: kedua anak memprediksi kelas yang sama. Sebanyak 8 dari 12 internal nodes adalah cosmetic.

**Hasil GridSearchCV.** Kombinasi terbaik yang ditemukan:

| Parameter | Nilai |
|-----------|:-----:|
| max_features | **5** |
| min_impurity_decrease | **0,001** |
| Best CV F1(0) | **0,8371** |

CV F1(0) meningkat dari 0,8165 (01-Tuned) menjadi 0,8371 — perbaikan genuine di generalisasi. Namun, performa pada test set tetap stagnan:

| Metrik | 01-Tuned | 03-Combined | Delta |
|--------|:--------:|:-----------:|:-----:|
| F1(0) | 0,8889 | 0,8889 | 0,0000 |
| Recall(0) | 0,8571 | 0,8571 | 0,0000 |
| Precision(0) | 0,9231 | 0,9231 | 0,0000 |
| AUC | 0,9239 | 0,9239 | 0,0000 |
| Depth | 3 | 4 | +1 |
| Leaves | 7 | 6 | -1 |
| Nodes | 13 | 11 | -2 |
| Features Used | 6/14 | 5/14 | -1 |
| Top-2% | 92,0% | 93,6% | +1,6% |
| Train Acc | 0,9691 | 0,9712 | +0,0021 |

Test F1(0) tetap 0,8889 — identik dengan 01-Tuned. Top-2% justru meningkat dari 92% menjadi 93,6%, artinya konsentrasi fitur tidak teratasi melainkan semakin parah. Hanya 2 leaves untuk kelas "Tidak Tepat" — model tidak bisa membedakan subtipe mahasiswa berisiko. Selain itu, proporsi cosmetic splits justru naik: dari 2/13 (15%) di 01-Tuned menjadi 4/8 (50%) di 03-Combined.

**Konvergensi tiga pendekatan.** Perbandingan keempat model (Baseline, 01-Tuned, 02-Engineered, 03-Combined) menunjukkan pola yang jelas:

| Metrik | Baseline | 01-Tuned | 02-Eng | 03-Combined |
|--------|:--------:|:--------:|:------:|:-----------:|
| **F1(0)** | 0,867 | **0,889** | **0,889** | **0,889** |
| Recall(0) | 0,929 | **0,857** | **0,857** | **0,857** |
| Precision(0) | 0,813 | **0,923** | **0,923** | **0,923** |
| AUC | 0,950 | 0,924 | 0,924 | 0,924 |
| Depth | 8 | **3** | 4 | 4 |
| Leaves | 24 | 7 | 9 | **6** |
| Nodes | 47 | 13 | 17 | **11** |
| Feats Used | 12 | 6 | 6 | **5** |
| Train Acc | 1,000 | **0,969** | **0,969** | 0,971 |

Tiga pendekatan tuning yang berbeda — pre-pruning (max_depth), feature engineering, dan combined tuning (max_features + min_impurity_decrease) — semuanya konvergen ke nilai F1(0) yang sama: **0,889**. Tidak ada satu pun pendekatan yang mampu melampaui angka ini.

**Kesimpulan: ceiling model telah tercapai.** Konvergensi tiga pendekatan tuning yang berbeda pada F1(0)=0,889 menandakan bahwa **ceiling performa untuk single Decision Tree pada dataset ini telah tercapai**. Faktor pembatas bukanlah konfigurasi hyperparameter, melainkan karakteristik data itu sendiri: hanya 68 sampel kelas minoritas (11,2%) dalam 608 total mahasiswa. Struktur pohon paling sederhana (depth=3, 7 leaves) sudah mampu mengekstrak seluruh sinyal yang tersedia, dan penambahan kompleksitas tidak memberikan perbaikan. Model **01-Tuned** (Pre-Pruning Tuning) tetap menjadi model terbaik karena memberikan F1(0) tertinggi (0,889) dengan kompleksitas terendah (depth=3).

## 4.5 Decision Rules dan Feature Importance Model Terbaik

Model terbaik (01-Tuned: max_depth=3, min_samples_leaf=10, criterion='gini') menghasilkan pohon keputusan dengan kedalaman 3 dan 7 daun. Pada bagian ini, struktur pohon diterjemahkan menjadi aturan keputusan yang dapat dipahami oleh pihak non-teknis, dan analisis feature importance digunakan untuk mengidentifikasi faktor dominan dalam prediksi ketepatan lulus.

Visualisasi pohon utuh telah disajikan pada Gambar 36 (subbab 4.4.1). Berikut adalah terjemahan setiap jalur dari akar ke daun dalam bentuk aturan logika:

| Rule | Path Keputusan | Prediksi | Jumlah Sampel (Train) |
|:----:|----------------|:--------:|:---------------------:|
| R1 | `sks_sem2 ≤ 18,5` → `failed_courses ≤ 0,5` → `ips_sem1 ≤ 3,01` | Tepat Waktu | 252 |
| R2 | `sks_sem2 ≤ 18,5` → `failed_courses ≤ 0,5` → `ips_sem1 > 3,01` | Tepat Waktu | 132 |
| R3 | `sks_sem2 ≤ 18,5` → `failed_courses > 0,5` | Tepat Waktu | 26 |
| R4 | `sks_sem2 > 18,5` → `sks_sem3 ≤ 18,5` → `ips_std ≤ 0,29` | **Tidak Tepat** | 28 |
| R5 | `sks_sem2 > 18,5` → `sks_sem3 ≤ 18,5` → `ips_std > 0,29` | **Tidak Tepat** | 18 |
| R6 | `sks_sem2 > 18,5` → `sks_sem3 > 18,5` → `avg_ips ≤ 3,31` | Tepat Waktu | 23 |
| R7 | `sks_sem2 > 18,5` → `sks_sem3 > 18,5` → `avg_ips > 3,31` | Tepat Waktu | 7 |

Tujuh aturan ini dapat disederhanakan menjadi **tiga aturan bisnis** utama:

**Aturan 1 — Jalur aman.** Mahasiswa dengan SKS semester 2 tidak berlebihan (`sks_sem2 ≤ 18,5`) diprediksi lulus tepat waktu, terlepas dari nilai IPS dan jumlah mata kuliah gagal. Aturan ini mencakup 410 mahasiswa (84% dari data latih).

**Aturan 2 — Overload lalu collapse (BERISIKO).** Mahasiswa dengan SKS semester 2 tinggi (`sks_sem2 > 18,5`) yang kemudian menurun drastis di semester 3 (`sks_sem3 ≤ 18,5`) diprediksi **tidak lulus tepat waktu**. Aturan ini mencakup 46 mahasiswa dan merupakan satu-satunya jalur yang menghasilkan prediksi negatif. Pola ini disebut **"overload then collapse"**: mahasiswa mengambil beban studi berlebihan di semester 2, lalu mengalami penurunan drastis di semester 3 — indikasi ketidakmampuan mempertahankan beban akademik.

**Aturan 3 — Overload berkelanjutan.** Mahasiswa dengan SKS semester 2 tinggi yang tetap mempertahankan SKS tinggi di semester 3 (`sks_sem3 > 18,5`) diprediksi lulus tepat waktu. Aturan ini mencakup 30 mahasiswa. Meskipun mengambil beban berat, mereka mampu mempertahankan kinerja, yang tercermin dari rata-rata IPS yang baik (`avg_ips ≤ 3,31`).

Distribusi feature importance berdasarkan Mean Decrease in Impurity (MDI) disajikan pada Tabel 4.1 dan visualisasinya pada Gambar 35 (subbab 4.4.1).

**Tabel 4.1** Feature Importance Model Terbaik (MDI)

| Rank | Fitur | MDI Importance | Kumulatif | Kategori |
|:----:|-------|:--------------:|:---------:|:--------:|
| 1 | `sks_sem2` | **0,5576** | 55,8% | Beban Studi |
| 2 | `sks_sem3` | **0,3614** | 91,9% | Beban Studi |
| 3 | `ips_std` | 0,0474 | 96,7% | Volatilitas IPS |
| 4 | `avg_ips` | 0,0172 | 98,4% | Rata-rata IPS |
| 5 | `failed_courses` | 0,0120 | 99,6% | Riwayat Gagal |
| 6 | `ips_sem1` | 0,0044 | 100,0% | Nilai Akademik |
| 7–14 | 8 fitur lainnya | **0,0000** | 100,0% | — |

Dua fitur SKS mendominasi: `sks_sem2` (55,76%) dan `sks_sem3` (36,14%) — total **91,9%** dari seluruh importance. **Delapan fitur** memiliki MDI nol: `ips_sem3`, `sks_sem1`, `ips_sem2`, `failed_in_sem1`, `ips_trend`, `repeated_courses`, `ips_min`, dan `sks_completion_ratio`. Artinya, model tidak menggunakan fitur-fitur tersebut sama sekali dalam pengambilan keputusan.

**Validasi dengan permutation importance.** Untuk memastikan hasil MDI tidak bias terhadap fitur kontinu dengan banyak nilai unik, dilakukan validasi menggunakan permutation importance (Gambar 53).

![Permutation Importance vs MDI](5-evaluation/01-evaluation-results_files/01-evaluation-results_18_2.png)

**Gambar 53.** Perbandingan MDI (kiri) dan permutation importance (kanan). MDI dan permutation importance sepakat pada urutan dua besar (`sks_sem2`, `sks_sem3`). Korelasi Spearman antara keduanya adalah 0,675 (p=0,008).

Permutation importance mengkonfirmasi dominasi `sks_sem2` (mean=0,128) dan `sks_sem3` (mean=0,066). Perbedaan utama: permutation importance memberikan nilai yang lebih tersebar dan tidak nol pada fitur dengan MDI=0, menunjukkan bahwa beberapa fitur mungkin memiliki kontribusi kecil namun tidak terdeteksi oleh MDI karena bias Gini impurity terhadap fitur kontinu.

**Korelasi Spearman** antara MDI dan permutation importance sebesar **0,675 (p=0,008)** — korelasi positif signifikan yang memvalidasi bahwa urutan kepentingan fitur dari kedua metode konsisten.

**Insight: Beban Studi vs Nilai Akademik**

Perbandingan kontribusi fitur SKS versus fitur IPS menghasilkan temuan yang jelas:

| Kategori Fitur | Total Importance | Jumlah Fitur |
|----------------|:---------------:|:------------:|
| **Beban Studi (SKS)** | **91,9%** | 3 fitur |
| Nilai Akademik (IPS) | 6,9% | 3 fitur |
| Riwayat Gagal | 1,2% | 1 fitur |
| Lainnya | 0,0% | 7 fitur |

**Beban studi (SKS) 13 kali lebih penting daripada nilai akademik (IPS)** dalam memprediksi ketepatan lulus. Temuan ini memiliki implikasi kebijakan yang penting: program studi sebaiknya lebih memfokuskan sistem early-warning pada pemantauan pola SKS mahasiswa daripada nilai akademik semata.

Fakta bahwa `sks_sem1` memiliki importance 0,0% sementara `sks_sem2` mendominasi di 55,8% menunjukkan bahwa **SKS semester 1 bukanlah indikator risiko** — hampir semua mahasiswa mengambil beban normal di semester pertama. Pola risiko baru terlihat pada semester 2, ketika mahasiswa mulai mengambil beban berlebihan yang kemudian menurun di semester 3 (pola "overload lalu collapse").

## 4.6 Evaluasi: Temporal Validation — Temuan Kritis

### Fase 1: Temporal Validation (Temuan Kritis)

Model terbaik (01-Tuned) yang mencapai F1(0)=0,889 pada stratified split diuji pada skenario temporal untuk menguji kemampuan generalisasi ke angkatan baru. Hasil pengujian menunjukkan **kegagalan total**: model tidak mendeteksi satu pun mahasiswa berisiko.

Tabel perbandingan seluruh metrik antara kedua pendekatan validasi:

**Tabel 4.2** Perbandingan Metrik Stratified vs Temporal

| Metrik | Stratified | Temporal | Delta |
|--------|:----------:|:--------:|:-----:|
| Accuracy | **0,9754** | 0,7662 | -0,2092 |
| Precision(0) | **0,9231** | 0,0000 | -0,9231 |
| Recall(0) | **0,8571** | 0,0000 | -0,8571 |
| F1(0) | **0,8889** | 0,0000 | -0,8889 |
| AUC (ROC) | **0,9550** | 0,7098 | -0,2452 |

Seluruh metrik kelas minoritas — Precision(0), Recall(0), dan F1(0) — bernilai **0,000** pada temporal validation. Model memprediksi semua 231 mahasiswa temporal test sebagai "Tepat Waktu". Confusion matrix temporal (Gambar 47) menunjukkan bahwa dari 54 mahasiswa berisiko, **tidak ada satu pun yang terdeteksi** (0 True Negative, 54 False Negative).

![Confusion Matrix Temporal](5-evaluation/01-evaluation-results_files/01-evaluation-results_4_2.png)

**Gambar 47.** Confusion matrix temporal validation. Model memprediksi seluruh 231 mahasiswa sebagai "Tepat Waktu" — 0 True Negative, 54 False Negative, 0 False Positive, 177 True Positive.

Visualisasi perbandingan metrik antar skenario validasi disajikan pada Gambar 46.

![Perbandingan Metrik Temporal vs Stratified](5-evaluation/01-evaluation-results_files/01-evaluation-results_4_1.png)

**Gambar 46.** Perbandingan metrik stratified vs temporal. Terdapat kesenjangan ekstrem pada Precision(0), Recall(0), dan F1(0) — semua turun dari >0,85 menjadi 0,000 pada temporal.

**Catatan AUC.** AUC stratified tercatat 0,9550 (ROC AUC dengan probabilitas) sementara pada subbab 4.4.1 AUC tercatat 0,9239 (dengan prediksi hard). Perbedaan ini disebabkan oleh metode komputasi: ROC AUC menggunakan probabilitas prediksi yang lebih granular, sedangkan AUC pada classification report dihitung dari prediksi biner.

Kegagalan temporal disebabkan oleh tiga faktor: (1) sample imbalance — temporal train hanya memiliki 14 sampel negatif (3,7%) berbanding 54 (11,1%) pada stratified train; (2) distribution shift — `sks_sem2` mean melonjak dari ~11 (2020–2021) menjadi ~19 (2023); (3) interaksi `min_samples_leaf=10` dengan sampel minoritas terbatas — tree tidak dapat membentuk leaf prediksi minoritas. Konsekuensinya, seluruh metrik stratified bersifat ilusif: model sejatinya memiliki F1(0)=0,000 pada skenario deployment.

### Fase 2: Deep Error Analysis

Seluruh 54 mahasiswa berisiko di temporal test tidak terdeteksi (false negative). Breakdown error menunjukkan pola yang jelas:

| Aspek | Rincian |
|-------|---------|
| **Program studi** | 96,3% FN berasal dari program S1 (IH) |
| **Angkatan** | 90,7% FN dari angkatan 2023 (49 dari 54 FN) |
| **Rule "overload-collapse"** | 52 dari 54 FN (96,3%) memenuhi aturan `sks_sem2 > 18,5 AND sks_sem3 ≤ 18,5` |
| **Presisi rule** | Seluruh 52 mahasiswa yang memenuhi rule tersebut benar-benar berisiko (presisi 100%) |
| **Angkatan 2022** | 5 negatif, 0 terdeteksi — semua FN |
| **Angkatan 2023** | 49 negatif, 0 terdeteksi — semua FN; akurasi pada angkatan ini hanya 2% |

Temuan kritis: aturan "overload lalu collapse" mencakup 96,3% mahasiswa berisiko di temporal test dan memiliki presisi sempurna. Model temporal gagal mendeteksi mereka, tetapi aturan bisnis ini sendiri dapat berfungsi sebagai sistem peringatan yang andal.

### Fase 3: Repeated Cross-Validation

Untuk mengatasi keterbatasan single split (hanya 14 sampel negatif di stratified test), dilakukan Repeated Stratified 10×10-Fold CV (100 evaluasi total):

| Metrik | Mean | Std | 95% CI Low | 95% CI High |
|--------|:----:|:---:|:----------:|:-----------:|
| F1(0) | **0,833** | 0,096 | 0,814 | 0,851 |
| Recall(0) | 0,783 | 0,135 | 0,757 | 0,810 |
| Precision(0) | 0,911 | 0,103 | 0,890 | 0,931 |
| AUC | 0,970 | 0,040 | 0,962 | 0,978 |

Single split recall(0)=0,857 **overestimates sebesar +7,4%** dibandingkan CV mean (0,783). Single split F1(0)=0,889 juga overestimates +5,6% dibandingkan CV mean (0,833). CV mean memberikan estimasi yang lebih realistis, namun tetap hanya berlaku untuk stratified scenario — tidak untuk temporal.

### Fase 4–6: Permutation Importance, Rule Stability, Distribution Shift

**Permutation importance** (Fase 4) telah dibahas pada subbab 4.5. MDI dan permutation importance sepakat bahwa `sks_sem2` dan `sks_sem3` adalah dua fitur dominan (Spearman r=0,675, p=0,008). Permutation importance mengkonfirmasi bahwa dominasi SKS bukan artefak bias MDI.

**Rule stability** (Fase 5). Root split `sks_sem2` muncul di 10 dari 10 CV folds (100%), mengkonfirmasi bahwa fitur ini stabil sebagai pemisah utama antar kelas pada skenario stratified. Namun, pada temporal split root split bergeser ke `failed_courses` — menunjukkan ketidakstabilan struktur keputusan antar cohort.

**Distribution shift** (Fase 6) dibahas secara mendalam pada subbab 4.7. Rata-rata `sks_sem2` naik dari ~11 pada angkatan training (2020–2021) menjadi ~19 pada angkatan test (2023), menyebabkan model yang dilatih pada data historis tidak dapat menggeneralisasi ke cohort baru.

## 4.7 Analisis Distribution Shift

Salah satu faktor kegagalan temporal yang teridentifikasi pada subbab 4.6 adalah perubahan distribusi fitur antar cohort (distribution shift). Analisis lebih mendalam terhadap pergeseran ini penting untuk memahami mengapa model yang dilatih pada data historis tidak dapat menggeneralisasi ke angkatan baru, serta untuk menentukan strategi mitigasi yang tepat.

Distribusi `sks_sem2` dan `sks_sem3` menunjukkan perubahan signifikan antar angkatan, sebagaimana disajikan pada Tabel 4.3.

**Tabel 4.3** Rata-rata SKS dan IPS per Angkatan

| Angkatan | n | sks_sem2 (mean) | sks_sem3 (mean) | avg_ips (mean) | Neg/Total |
|:--------:|:-:|:---------------:|:---------------:|:--------------:|:---------:|
| 2015 | 116 | 17,33 | 10,73 | 3,23 | 2/116 |
| 2016 | 54 | 9,52 | 13,52 | 3,29 | 2/54 |
| 2017 | 48 | 20,83 | 15,92 | 3,33 | 2/48 |
| 2018 | 46 | 18,02 | 12,52 | 3,32 | 1/46 |
| 2019 | 27 | 18,59 | 19,00 | 3,37 | 0/27 |
| 2020 | 40 | **10,47** | 8,25 | 3,42 | 4/40 |
| 2021 | 46 | **11,37** | 9,43 | 3,40 | 3/46 |
| **2022** | 181 | **17,64** | 9,10 | 2,71 | 5/181 |
| **2023** | 50 | **19,20** | **17,78** | 2,93 | **49/50** |

Pola yang terlihat:

**Periode training (2015–2021):** `mean sks_sem2` bervariasi antara 9,5–20,8 dengan kecenderungan menurun pada angkatan 2020–2021 (~10–11). Proporsi mahasiswa berisiko relatif rendah (0–4 dari total per angkatan).

**Periode test (2022–2023):** `mean sks_sem2` melonjak drastis menjadi 17,6 (2022) dan 19,2 (2023) — kembali ke level angkatan 2017–2018 setelah sempat turun di 2020–2021. Lonjakan paling ekstrem terjadi pada `sks_sem3` di angkatan 2023: dari ~9 (2020–2022) menjadi 17,8 — hampir dua kali lipat.

**Fenomena angkatan 2023:** Sebanyak 49 dari 50 mahasiswa angkatan 2023 masuk kelas "Tidak Tepat Waktu". Pola SKS mereka sangat berbeda dari angkatan sebelumnya: `sks_sem2` tinggi (19,2) *dan* `sks_sem3` tetap tinggi (17,8), yang seharusnya masuk Aturan 3 (overload berkelanjutan → Tepat Waktu). Namun, kenyataannya hampir seluruh angkatan ini belum lulus — kemungkinan besar karena mereka masih dalam masa studi pada saat data diambil, bukan karena kegagalan akademik. Data ini mengindikasikan adanya **label leakage temporal**: sebagian besar angkatan 2023 belum menyelesaikan studi sehingga secara otomatis berlabel "Tidak Tepat", meskipun pola SKS mereka mirip dengan mahasiswa tepat waktu di angkatan sebelumnya.

Distribution shift memiliki tiga dampak langsung terhadap kemampuan generalisasi model:

1. **Perubahan distribusi fitur mengubah struktur keputusan.** Model stratified menggunakan `sks_sem2 ≤ 18,5` sebagai root split. Pada angkatan 2020–2021 dengan `sks_sem2` rendah (~11), threshold ini mudah dipenuhi sehingga sebagian besar mahasiswa masuk jalur aman. Namun pada angkatan 2022–2023 dengan `sks_sem2` tinggi (~18-19), threshold yang sama justru mengirim mayoritas mahasiswa ke cabang risiko — tetapi model temporal tidak memiliki leaf negatif, sehingga semua diprediksi "Tepat Waktu" (false negative).

2. **Root split berubah antar cohort.** Pada model temporal, root split bergeser dari `sks_sem2` menjadi `failed_courses`. Ini menunjukkan bahwa struktur keputusan yang dipelajari pada data historis tidak stabil — fitur yang paling informatif pada satu periode belum tentu berlaku pada periode berikutnya.

3. **Akurasi tinggi menutupi kegagalan.** Model temporal mencapai akurasi 76,62% hanya dengan memprediksi semua mahasiswa sebagai "Tepat Waktu". Namun, pada angkatan 2023 yang memiliki 98% mahasiswa berisiko, akurasi anjlok menjadi 2% — model tidak berguna sama sekali untuk cohort dengan karakteristik berbeda.

Distribution shift bersifat inheren pada data longitudinal dan tidak dapat dihilangkan sepenuhnya. Beberapa strategi mitigasi yang direkomendasikan:

**Periodic retraining.** Model harus dilatih ulang secara berkala (setiap semester atau setiap tahun) dengan menambahkan data angkatan terbaru. Hal ini memungkinkan model menangkap perubahan distribusi fitur secara bertahap. Retraining juga mengatasi masalah label leakage temporal: ketika angkatan 2023 akhirnya menyelesaikan studi, label aktual mereka dapat digunakan untuk training selanjutnya.

**Monitoring distribusi.** Implementasikan pipeline monitoring yang secara otomatis mendeteksi pergeseran distribusi fitur pada data baru. Jika `sks_sem2` mean bergeser lebih dari satu standar deviasi dari data training, model harus di-retrain sebelum digunakan untuk prediksi.

**Hybrid system.** Kombinasikan model ML dengan aturan bisnis tetap. Aturan "overload lalu collapse" (`sks_sem2 > 18,5 AND sks_sem3 ≤ 18,5`) terbukti memiliki presisi sempurna pada temporal test — seluruh 52 mahasiswa yang memenuhi aturan ini benar-benar berisiko. Aturan ini dapat digunakan sebagai fallback alert ketika model ML gagal mendeteksi mahasiswa berisiko karena keterbatasan data training.

## 4.8 Perbandingan Kinerja Antar Iterasi

Setelah seluruh iterasi pemodelan selesai dilakukan, perbandingan komprehensif antar konfigurasi diperlukan untuk memetakan perkembangan performa dan mengidentifikasi pola yang konsisten. Tabel 4.4 menyajikan perbandingan enam skenario pemodelan yang mencakup tiga iterasi dan dua strategi split.

**Tabel 4.4** Perbandingan Kinerja Seluruh Iterasi Pemodelan

| Iterasi | Split | Model | F1(0) | Recall(0) | Precision(0) | AUC | Depth | Leaves |
|:-------:|:-----:|-------|:-----:|:---------:|:------------:|:---:|:-----:|:------:|
| Iterasi 1 | Temporal | Baseline | 0,071 | 0,037 | 1,000 | 0,519 | 9 | 21 |
| Iterasi 1 | Stratified | Baseline | 0,765 | 0,929 | 0,650 | 0,932 | 8 | 23 |
| Iterasi 2 | Temporal | Global Median | 0,105 | 0,056 | 1,000 | 0,528 | 9 | 21 |
| Iterasi 2 | Stratified | Global Median | 0,867 | 0,929 | 0,813 | 0,950 | 8 | 24 |
| Iterasi 3 | Stratified | Pre-Pruning Tuned | **0,889** | 0,857 | **0,923** | 0,924 | **3** | **7** |
| Iterasi 3 | Stratified | Combined Tuned | **0,889** | 0,857 | 0,923 | 0,924 | 4 | 6 |

**Pola yang terlihat dari tabel perbandingan:**

**Kesenjangan temporal vs stratified konsisten di semua iterasi.** Pada Iterasi 1, Recall(0) temporal hanya 0,037 berbanding stratified 0,929 — kesenjangan 0,892 poin. Pada Iterasi 2, Recall(0) temporal naik tipis menjadi 0,056 namun stratified tetap di 0,929. Kesenjangan ini tidak pernah terselesaikan karena akar masalahnya bukan pada preprocessing atau hyperparameter, melainkan pada jumlah sampel minoritas yang tersedia untuk training temporal (14 negatif) versus stratified (54 negatif).

**Global median memperbaiki presisi stratified.** Perbandingan Iterasi 1 → Iterasi 2 pada stratified split menunjukkan Precision(0) meningkat dari 0,650 menjadi 0,813 (+0,163) tanpa mengorbankan Recall(0). F1(0) naik dari 0,765 menjadi 0,867 (+0,102). Perbaikan ini sepenuhnya berasal dari penghapusan temporal proxy melalui imputasi global median — bukan dari tuning hyperparameter.

**Hyperparameter tuning menyederhanakan model secara drastis.** Perbandingan Iterasi 2 → Iterasi 3 pada stratified split menunjukkan pengurangan kompleksitas yang sangat signifikan: depth turun dari 8 ke 3 (−62,5%), leaves dari 24 ke 7 (−70,8%), dan fitur yang digunakan dari 12 ke 6 (−50,0%). Meskipun Recall(0) sedikit menurun (0,929 → 0,857), Precision(0) meningkat drastis (0,813 → 0,923) sehingga F1(0) naik dari 0,867 menjadi 0,889.

**Ceiling F1(0)=0,889 tidak dapat ditembus.** Pre-Pruning Tuned (depth=3), Feature Engineering, dan Combined Tuning (max_features=5, min_impurity_decrease=0,001) semuanya konvergen ke F1(0)=0,889 yang sama. Tidak ada konfigurasi hyperparameter yang mampu melampaui batas ini, mengonfirmasi bahwa ceiling untuk single Decision Tree pada dataset ini telah tercapai.

**Precision(0) temporal sempurna namun tidak berguna.** Precision(0)=1,000 pada temporal split di Iterasi 1 dan 2 bukanlah indikator kualitas — melainkan konsekuensi dari model yang terlalu konservatif. Model hanya membuat prediksi negatif pada kasus yang sangat yakin (2 TN pada Iterasi 1, 3 TN pada Iterasi 2), dan tidak pernah salah. Namun Recall(0) yang mendekati nol (0,037–0,056) berarti model hampir tidak pernah mendeteksi mahasiswa berisiko — Precision sempurna menjadi tidak bermakna.

**Implikasi utama.** Seluruh metrik yang dilaporkan pada stratified split — termasuk F1(0)=0,889 dari model terbaik — **tidak dapat digunakan sebagai estimasi performa deployment**. Temporal validation telah membuktikan bahwa performa aktual model pada skenario nyata adalah F1(0)=0,000. Repeated CV memberikan estimasi yang lebih realistis (F1(0) mean=0,833) namun tetap tidak mendeteksi distribution shift antar cohort. Kesimpulan tegas dari perbandingan ini: **validasi temporal adalah gold standard** yang wajib digunakan untuk dataset longitudinal dengan karakteristik cohort yang berubah antar waktu.

---

# Bab V - Kesimpulan dan Saran

## 5.1 Kesimpulan

Penelitian ini berhasil membangun model klasifikasi Decision Tree untuk memprediksi ketepatan lulus mahasiswa menggunakan pendekatan CRISP-DM dengan tiga iterasi pemodelan bertahap. Berdasarkan seluruh hasil analisis dan pembahasan yang telah diuraikan, dapat ditarik enam kesimpulan utama sebagai berikut:

**1. Model Decision Tree berhasil dibangun dengan arsitektur yang sederhana dan interpretable.** Model terbaik diperoleh pada Iterasi 3 (Pre-Pruning Tuning) dengan konfigurasi `max_depth=3`, `min_samples_leaf=10`, dan `criterion='gini'`. Model memiliki kedalaman 3 level dengan 7 daun dan 13 node, menggunakan hanya 6 dari 14 fitur yang tersedia. Performa pada stratified split mencapai F1(0)=0,889, Recall(0)=0,857, dan Precision(0)=0,923 — metrik yang memenuhi seluruh target modeling plan. Overfitting berhasil dihilangkan: train accuracy (0,969) lebih rendah dari test accuracy (0,975), mengindikasikan model mampu menggeneralisasi pada data baru dalam skenario stratified.

**2. Fitur beban studi (SKS) jauh lebih determinan daripada nilai akademik (IPS) dalam memprediksi ketepatan lulus.** Dua fitur SKS mendominasi feature importance: `sks_sem2` (55,8%) dan `sks_sem3` (36,1%) — total 91,9% dari seluruh importance. Sebagai perbandingan, seluruh fitur IPS hanya menyumbang 6,9%, dan riwayat mata kuliah gagal hanya 1,2%. **Beban studi 13 kali lebih penting daripada nilai akademik.** Temuan ini dikonfirmasi oleh permutation importance yang menunjukkan korelasi Spearman sebesar 0,675 (p=0,008) dengan MDI, memvalidasi bahwa dominasi SKS bukan artefak bias algoritma.

**3. Aturan keputusan "overload lalu collapse" teridentifikasi sebagai pola risiko paling kuat.** Rule ini didefinisikan sebagai `sks_sem2 > 18,5 AND sks_sem3 ≤ 18,5` — mahasiswa yang mengambil SKS sangat tinggi di semester 2 (di atas 18,5) kemudian mengalami penurunan drastis di semester 3. Pada temporal test, rule ini mencakup 52 dari 54 mahasiswa berisiko (96,3%) dan memiliki **presisi 100%** — seluruh mahasiswa yang memenuhi rule ini benar-benar tidak lulus tepat waktu. Aturan ini dapat digunakan sebagai sistem peringatan dini berbasis aturan sederhana yang tidak memerlukan model machine learning.

**4. Temuan kritis: model gagal total pada temporal validation (F1(0)=0,000).** Model terbaik yang mencapai F1(0)=0,889 pada stratified split tidak mendeteksi satu pun dari 54 mahasiswa berisiko ketika diuji pada skenario temporal yang mensimulasikan kondisi deployment nyata (data latih angkatan ≤2021, data uji angkatan >2021). Confusion matrix menunjukkan seluruh 54 mahasiswa berisiko berada di kuadran false negative, sementara akurasi 76,62% tercapai semata-mata karena model benar mengklasifikasikan 177 mahasiswa tepat waktu. **Performa stratified adalah ilusi** — model sejatinya tidak berguna untuk skenario deployment sesungguhnya.

**5. Tiga iterasi modeling menunjukkan ceiling F1(0)=0,889 untuk single Decision Tree.** Tiga pendekatan tuning yang berbeda — pre-pruning (max_depth=3), feature engineering (4 fitur baru), dan combined tuning (max_features + min_impurity_decrease) — semuanya konvergen ke nilai F1(0)=0,889 yang identik. Tidak ada konfigurasi hyperparameter yang mampu melampaui batas ini. Faktor pembatas bukanlah konfigurasi model, melainkan karakteristik data itu sendiri: hanya 68 sampel kelas minoritas (11,2%) dalam 608 total mahasiswa. Repeated Stratified 10×10-Fold CV memberikan estimasi yang lebih konservatif dengan F1(0) mean=0,833 [95% CI: 0,814–0,851], mengonfirmasi bahwa single split stratified overestimates F1(0) sebesar +5,6%.

**6. Model dinyatakan tidak layak deployment.** Tiga faktor penyebab kegagalan temporal saling terkait dan tidak dapat diatasi hanya melalui hyperparameter tuning: (a) **sample imbalance** — temporal train hanya memiliki 14 sampel negatif (3,7%) yang tidak cukup untuk membentuk leaf prediksi minoritas dengan constraint `min_samples_leaf=10`; (b) **distribution shift** — rata-rata `sks_sem2` melonjak dari ~11 pada angkatan training (2020–2021) menjadi ~19 pada angkatan test (2023), mengubah karakteristik fitur secara fundamental; (c) **label leakage temporal** — 49 dari 50 mahasiswa angkatan 2023 berlabel "Tidak Tepat" karena belum menyelesaikan studi, bukan karena kegagalan akademik. Tanpa penanganan class imbalance, periodic retraining, atau pendekatan rule-based, model tidak dapat diandalkan untuk memprediksi mahasiswa angkatan baru.

## 5.2 Saran

Berdasarkan kesimpulan dan temuan kritis selama penelitian, diajukan sejumlah saran yang dikelompokkan menjadi tiga kategori: saran untuk modeling lanjutan, saran untuk institusi, dan saran untuk penelitian selanjutnya.

**Saran untuk Modeling Lanjutan:**

**1. Terapkan SMOTE atau `class_weight='balanced'` pada temporal split.** Akar utama kegagalan temporal adalah ketidakseimbangan kelas ekstrem di data latih (14 negatif dari 377 mahasiswa, 3,7%). SMOTE (Synthetic Minority Oversampling Technique) dapat digunakan untuk membuat sampel sintetis kelas minoritas melalui interpolasi antar sampel negatif terdekat, sehingga jumlah sampel negatif di temporal train meningkat ke tingkat yang memungkinkan Decision Tree membentuk leaf prediksi minoritas. Alternatif yang lebih sederhana adalah `class_weight='balanced'` yang memberikan bobot lebih tinggi pada kelas minoritas saat menghitung fungsi loss. Namun, eksperimen pada Iterasi 3 menunjukkan bahwa `class_weight='balanced'` justru menyebabkan overfitting pada dataset dengan sampel minoritas sangat terbatas — sehingga SMOTE yang bekerja pada level data (bukan level algoritma) mungkin lebih efektif.

**2. Eksplorasi ensemble methods (Random Forest, Gradient Boosting).** Single Decision Tree memiliki keterbatasan varians tinggi dan ceiling performa yang telah terbukti (F1(0)=0,889). Random Forest mengatasi kelemahan ini dengan membangun banyak pohon pada subset data dan fitur yang berbeda, kemudian merata-ratakan prediksi. Gradient Boosting (XGBoost, LightGBM) membangun pohon secara sekuensial dengan setiap pohon baru memperbaiki kesalahan pohon sebelumnya. Kedua pendekatan ini berpotensi meningkatkan stabilitas prediksi dan menembus ceiling F1(0)=0,889 yang saat ini membatasi single Decision Tree.

**3. Pisahkan model per program studi.** Karakteristik akademik program vokasi (D3) dan sarjana (S1) berbeda secara fundamental: struktur kurikulum D3 lebih terstruktur dengan 3 tahun masa studi, sementara S1 memiliki fleksibilitas lebih tinggi dengan 4 tahun masa studi. Data menunjukkan proporsi ketidaktepatan lulus di S1 (12,6%) hampir dua kali lipat dari D3 (6,8%). Model terpisah per program studi memungkinkan Decision Tree menangkap pola spesifik yang mungkin berbeda antara kedua kelompok, seperti threshold SKS yang optimal atau kombinasi fitur yang relevan. Meskipun fitur `program` memiliki near-zero importance pada model global, pemisahan model dapat mengungkap interaksi yang lebih kompleks yang tidak terdeteksi oleh satu pohon global.

**4. Kumpulkan lebih banyak data historis.** Temporal train saat ini hanya mencakup angkatan 2015–2021 dengan 14 sampel negatif. Penambahan data angkatan sebelum 2015 (jika tersedia) atau penggabungan dengan data dari program studi lain di institusi yang sama dapat memperkaya jumlah sampel minoritas. Setiap tambahan sampel negatif di data latih memiliki dampak langsung terhadap kemampuan model membentuk leaf prediksi minoritas — terutama dengan constraint `min_samples_leaf` yang diperlukan untuk mencegah overfitting.

**5. Pertimbangkan hybrid system: rule-based alert + ML.** Aturan "overload lalu collapse" (`sks_sem2 > 18,5 AND sks_sem3 ≤ 18,5`) terbukti memiliki **presisi 100%** pada temporal test — seluruh 52 mahasiswa yang memenuhi aturan ini benar-benar berisiko, namun tidak satu pun terdeteksi oleh model ML. Sistem hibrida dapat menggunakan aturan ini sebagai first-level alert yang tidak memerlukan training data, dan model ML sebagai second-level filter untuk mahasiswa yang tidak tercakup oleh aturan. Pendekatan ini menggabungkan keandalan rule-based (presisi sempurna) dengan fleksibilitas ML.

**6. Periodic retraining setiap tahun.** Distribution shift antar cohort adalah fenomena natural pada data longitudinal. Model yang dilatih pada data historis tahun tertentu tidak dapat diharapkan berlaku sama untuk mahasiswa angkatan berikutnya. Strategi retraining tahunan dengan menambahkan data angkatan terbaru ke dalam training set memungkinkan model menangkap perubahan distribusi fitur secara bertahap. Retraining juga mengatasi masalah label leakage temporal: ketika angkatan 2023 akhirnya menyelesaikan studi, label aktual mereka dapat digunakan untuk training selanjutnya, menggantikan label sementara "Tidak Tepat" yang saat ini menyesatkan.

**Saran untuk Institusi:**

**1. Implementasikan early-warning system berbasis rule SKS semester 2–3.** Sebelum model ML siap deployment, institusi dapat segera menerapkan sistem peringatan dini sederhana berdasarkan aturan yang telah teridentifikasi: mahasiswa dengan `sks_sem2 > 18,5` dan `sks_sem3 ≤ 18,5` harus diberikan perhatian khusus oleh dosen wali. Aturan ini mudah diimplementasikan sebagai query SQL pada database akademik dan tidak memerlukan infrastruktur machine learning. Dengan presisi 100%, aturan ini tidak menghasilkan false positive — sumber daya intervensi dapat difokuskan secara efisien.

**2. Evaluasi kebijakan beban SKS per semester.** Temuan bahwa beban studi (SKS) 13 kali lebih penting daripada nilai akademik (IPS) dalam memprediksi ketepatan lulus memiliki implikasi kebijakan yang signifikan. Institusi perlu mengevaluasi apakah kebijakan pengambilan SKS maksimum di semester awal sudah tepat, atau justru mendorong mahasiswa untuk overload yang berujung pada keterlambatan lulus. Pola "overload lalu collapse" yang teridentifikasi menunjukkan bahwa mahasiswa yang mengambil SKS sangat tinggi di semester 2 (di atas 18,5) kemudian mengalami penurunan drastis di semester 3 adalah kelompok risiko tertinggi. Pembatasan SKS maksimum di semester awal atau sistem approval untuk SKS di atas threshold tertentu dapat menjadi langkah preventif.

**3. Tingkatkan kualitas pencatatan data akademik.** Penelitian ini menemukan tiga masalah kualitas data yang signifikan: (a) placeholder IPS=0.0 pada 219 mahasiswa (36%) yang bukan merupakan nilai aktual; (b) bug SKS kumulatif pada angkatan 2020+ akibat migrasi vendor database; (c) missing values pada data IPS dan SKS di angkatan awal. Perbaikan kualitas pencatatan data sejak hulu akan sangat membantu penelitian serupa di masa mendatang. Standarisasi format penyimpanan data antar semester dan angkatan, validasi input pada sistem akademik, serta dokumentasi riwayat migrasi database adalah langkah-langkah konkret yang dapat dilakukan.

**Saran untuk Penelitian Selanjutnya:**

**1. Tambahkan fitur non-akademik.** Penelitian ini hanya menggunakan fitur akademik (IPS, SKS, nilai mata kuliah) dari tiga semester pertama. Penelitian selanjutnya dapat mempertimbangkan fitur non-akademik yang mungkin berkontribusi terhadap ketepatan lulus, seperti data demografi (usia, jenis kelamin, daerah asal), data ekonomi (status beasiswa, pekerjaan orang tua), jalur masuk (SNMPTN, SBMPTN, mandiri), atau data psikologis (hasil tes minat bakat). Fitur-fitur ini dapat memberikan sinyal tambahan yang tidak tercakup oleh performa akademik saja.

**2. Eksplorasi algoritma lain.** Decision Tree dipilih dalam penelitian ini karena interpretabilitasnya. Penelitian selanjutnya dapat membandingkan Decision Tree dengan algoritma lain seperti XGBoost, LightGBM, atau Random Forest yang mungkin memiliki performa lebih tinggi meskipun interpretabilitasnya lebih rendah. Jika jumlah data mencukupi, deep learning juga dapat dieksplorasi — meskipun untuk dataset dengan 608 sampel, deep learning mungkin belum memberikan hasil optimal dan berisiko overfitting.

**3. Uji model pada cohort mendatang.** Validasi temporal dalam penelitian ini menggunakan data angkatan 2022–2023 sebagai test set. Untuk memvalidasi temuan secara longitudinal, penelitian selanjutnya perlu menguji model pada angkatan 2024 dan seterusnya — cohort yang sepenuhnya berada di luar periode data yang digunakan dalam penelitian ini. Pengujian pada cohort mendatang akan memberikan konfirmasi definitif mengenai kemampuan generalisasi model dan relevansi aturan keputusan yang telah diidentifikasi.
