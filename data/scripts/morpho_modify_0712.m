function out1  = morpho_modify_0712(in1)

DE_c  = bwmorph(in1,'close')   ;
DE_sk = bwmorph(DE_c,'skel',100);
DE_sh = DE_sk ; 
DE_sh = bwmorph(DE_sk,'shrink') ;
DE_cl = bwmorph(DE_sh,'clean')  ;
DE_hb = bwmorph(DE_cl,'hbreak',5000) ;
DE_sp =DE_hb ; 
DE_sp = bwmorph(DE_hb,'spur',2) ;
DE_cl2 = bwmorph(DE_sp,'clean') ;


% added on july 12 2016
se = strel('disk',2);
closeBW = imclose(DE_cl2,se) ;
DE11 =  bwmorph(closeBW,'thin',10);


out1 = DE11 ;


end