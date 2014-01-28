live_or_studio
==============
>**Designer:** Junbo Zhao, Wuhan University, Working at Douban Inc.      
**Email:** zhaojunbo1992chasing@gmail.com	      +86-18672365683      

Introduction
-----------------------------------
  This project is mainly about a real classification problem on music. Are the two versions of music, live and studio, can be 
  classified? Tracing from that, we mainly apply the famous MFCC feature and two sorts of SVM to finish this problem. Note that
  our SVM training data is not simply traditional MFCC, but statistically processed MFCC which has been proved that accelerate 
  both training and testing procedures.      
  The experimental results show that our method yield and 93%+ precision on this classification problem, Live or Studio?!   
  
### MFCC
  Widely used in speech community and you can find the specific info at: http://en.wikipedia.org/wiki/Mel-frequency_cepstrum    
  
### Two SVM Frameworks
  Firstly, we adopt LIB-SVM of python version to build our baseline framework, which is from: http://www.csie.ntu.edu.tw/~cjlin/libsvm/     
  However, not only apply this package in a traditional way, we have exploited it following the idea of Pro. Efros, Ensembled Exemplar SVMs:          
  http://www.cs.cmu.edu/~tmalisie/projects/iccv11/      
  This method has been proved to be robust on some multi-class vision tasks, like PASCAL VOC Challenge. But it yields great result on this binary
  class problem.
  
### Dpark
  Dpark, as an open-source tool cloned from Spark, is widely known nowadays in engineering community. It is produced at Douban Inc., where I am currently working for, and 
  helps us extremely accelerate the scripts. It has friendly python intefaces, and is very convenient to use. You can find it on this github repo:        
  https://github.com/douban/dpark

To get start
-----------------------------------
1. Download LIBSVM from the website above.     
2. Get to know something about Dpark and install it on your computer or server.    
3. It is highly recommended to use "mesos", if it is available to you.     
4. Extract MFCC and statistically process the features. But bear in mind that original MFCC can also be used under our framework.      
5. Run and compare the results of two frameworks!


Platform
-----------------------------------
We implement our frameworks by python 2.7.5, on Ubuntu 64-bit server. This project should be compatibled on other platforms. If you got some problems, feel free to contact me.
        

    
