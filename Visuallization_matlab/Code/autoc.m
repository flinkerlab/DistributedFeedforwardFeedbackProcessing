load('/Users/ranwang/Documents/writen_paper/NER2020/entiregrid_ecog_742prod.mat');
autoc_ecog = zeros(50,15,15,144);
for i=1:15
    for j=1:15
        for s=1:50
            autoc_ecog(s,i,j,:) = autocorr(squeeze(ecog(s,:,:,i,j)),143);
        end
    end
end
autoc_ecog = squeeze(mean(autoc_ecog));
figure;plot([1:144]/125*1000,squeeze(autoc_ecog(5,13,:))); xlabel('Lag (ms)');ylabel('correlation');title('STG electrode');


load('/Users/ranwang/Documents/writen_paper/NER2020/samples/742_wave.mat');
autoc_wav = zeros(50,16384);
for s=1:50
    autoc_wav(s,:) = autocorr(squeeze(wav(s,1,:)),16383);
end
autoc_wav = squeeze(mean(autoc_wav));
figure;plot([1:16384]/16000*1000,squeeze(autoc_wav)); xlabel('Lag (ms)');ylabel('correlation');title('wav');


load('/Users/ranwang/Documents/writen_paper/NER2020/samples/742_spec.mat');
autoc_spec = zeros(50,256,128);
for s=1:50
    for c=1:256
        autoc_spec(s,c,:) = autocorr(squeeze(spec(c,s,:)),127);
    end
end
autoc_spec = squeeze(mean(mean(autoc_spec)));
figure;plot([1:128]/125*1000,squeeze(autoc_spec)); xlabel('Lag (ms)');ylabel('correlation');title('spec');


xc_ecog = zeros(50,256,255);
i=5;j=13;
for s=1:50
    for c=1:256
        xc_ecog(s,c,:) = xcorr(squeeze(ecog(s,1:128,:,i,j)),squeeze(spec(c,s,:))',127);
    end
end
xc_ecog = squeeze(mean(mean(xc_ecog)));
figure;plot([-127:127]/125*1000,xc_ecog); xlabel('Lag (ms)');ylabel('correlation');title('STG electrode vs spec');
