Otonom B2B Risk ve Aksiyon Platformu 

1. Çözdüğümüz Problem 

KOBİ’lerin ve toptan B2B satış yapan işletmelerin en büyük sorunu veriyi toplayamamak değil, biriken veriden zamanında anlam çıkaramamaktır. Bir müşterinin siparişleri aydan aya sessizce düştüğünde (Churn) veya artan maliyetler karşısında satış fiyatı sabit kalıp kâr eridiğinde (Margin Squeeze), işletme sahipleri bunu aylar sonra bilançoda fark etmektedir. KOBİ'lerin veri analisti veya finansal danışman tutacak bütçeleri yoktur; acil durumlarda onlara ne yapmaları gerektiğini söyleyecek proaktif sistemlere ihtiyaçları vardır. 

2. KOBİ-Zeka Nedir ve Nasıl Çözer? 

KOBİ-Zeka, işletmenin mevcut satış veritabanını sürekli izleyen ve anormallikleri tespit ettiğinde yöneticiye "hazır aksiyonlar" sunan otonom bir ajan sistemidir. 

Sistem sıradan bir sohbet botu (chatbot) değildir. Arka planda birbirini denetleyen üç farklı yapay zeka ajanı çalışır: 

Analist Ajan: Veriyi tarar ve kârı eriyen/kaybedilmek üzere olan müşterileri matematiksel olarak tespit eder. 

Strateji Ajanı: Tespit edilen soruna göre ticari bir çözüm üretir (Örn: "Maliyeti kurtarmak için %5 iskonto ile peşin ödemeye ikna et"). 

Aksiyon Ajanı: Bu stratejiyi, yöneticinin müşteriye tek tıkla gönderebileceği profesyonel bir WhatsApp mesajına dönüştürür. 

3. Yenilikçi Yönümüz ve Farkımız 

"Sohbet" Değil, "Aksiyon" Arayüzü: İşletme sahiplerinin yapay zekayla uzun uzun sohbet edecek vakti yoktur. Bu yüzden sistemi bir "Aksiyon Gelen Kutusu" olarak tasarladık. Riskler, renkli etiketler ve grafiklerle kartlar halinde sunulur. Yönetici sadece onaylar ve tek tıkla WhatsApp üzerinden müşteriye ulaşır. 

Sıfır Veri İhlali (Veri Maskeleme): B2B projelerindeki en büyük handikap olan veri gizliliğini çözdük. Sistem, veritabanındaki gerçek müşteri isimlerini ve hassas bilgileri yapay zekaya (LLM) göndermeden önce anlık olarak maskeler (Örn: "CUSTOMER_1"). LLM sadece metrikleri işler, gerçek isimler yalnızca kullanıcının kendi ekranında, arayüzde tekrar eşleştirilir. 

Ajan İş Akışı (Agentic Workflow): Tek bir komut-cevap döngüsü yerine, karmaşık işleri alt parçalara bölen "LangGraph" mimarisi kullandık. Bu sayede sistemin halüsinasyon görme veya yanlış ticari tavsiye verme ihtimalini minimuma indirdik. 

4. Teknik Mimari 

Bilişsel Motor: Hız ve maliyet optimizasyonu için Google Gemini 1.5 Flash API. 

Orkestrasyon: Ajanların yönlendirmesi ve durum yönetimi için LangGraph ve LangChain. 

Veritabanı ve Arayüz: Kurumsal kullanıma uygun PostgreSQL entegrasyonu ve hızlı prototipleme için Streamlit (Kullanıcı durum yönetimi - session state ile desteklenmiş B2B SaaS görünümü). 

5. Sonuç 

KOBİ-Zeka ile bir işletme sahibinin saatlerini alacak veri analizi ve müşteri iletişim süreci saniyelere inmektedir. Otonom yapısı sayesinde işletmeler riski krize dönüşmeden fark eder ve aksiyon alır. 
