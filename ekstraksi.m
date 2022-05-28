clear
cd('D:\KULIAH\SEMESTER 6\COMVIS\ekstraksi\bunga');
datasetku={'b1';'b2'};
jmlkelas=length(datasetku);
for n=1:jmlkelas
    cd(char(datasetku(n)));
    datacitra=dir('*.jpg');
    jmldata=length(datacitra);
    for i=1:jmldata
        namefile=datacitra(i).name;
        citrai=rgb2gray(imread(namefile));
        fitur=graycoprops(graycomatrix(citrai));%glcm
        fitur_mat(i+jmldata*(n-1),1)=fitur.Contrast;
        fitur_mat(i+jmldata*(n-1),2)=fitur.Correlation;
        fitur_mat(i+jmldata*(n-1),3)=fitur.Energy;
        fitur_mat(i+jmldata*(n-1),4)=fitur.Homogeneity;
        
        kelas(i+jmldata*(n-1))=n;
    end
    cd('..');
end

%pengujian
model=fitcknn(fitur_mat, kelas');%model knn
cd('tes')
for j=1:jmlkelas
    nama=sprintf('b%d.jpg',j);
    a=rgb2gray(imread(nama));
    m=graycomatrix(a);
    g=graycoprops(m);
    uji(j,1)=g.Contrast;
    uji(j,2)=g.Correlation;
    uji(j,3)=g.Energy;
    uji(j,4)=g.Homogeneity;
    
    target(j)=j;
    klasifikasi(j)=model.predict(uji(j,:)); %melakukan prediksi dari model
    if klasifikasi(j)==target(j)
        hasil(j)={'Benar'}
    else
        hasil(j)={'Salah'}
    end
end
[{'Contrast','Correlation','Energy','Homogeneity','Target','Kelas','Hasil'};
    num2cell([uji target' klasifikasi']) hasil']

cm=confusionmat(target', klasifikasi')
akurasi=sum(diag(cm))/sum(sum(cm))