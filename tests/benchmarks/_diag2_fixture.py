"""Shared physical-artifact helpers for DIAG2 contract tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Callable

from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    canonical_json_bytes,
)

DIAG2_TEST_NONCE = "0" * 32
_DIAG2_ARCHIVED_FROZEN_SOURCE = {
    "src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py": (
        "c28a598a56eae109b3e61f846ae58c34b97a2cdc5fe92fdb15af0a668eb380de",
        (
            b"c-rkfYj4}evfuS92!F^W6Iyw+J?CDdDyrH_t0J{sTM5ux6a+2NHghCWB`GH^ivISyZ|-CELD}u;xrl+pCOJDhJ3B"
            b"i&J2Sg`aBy%^+?C7XuqoPdRa9NHZR+>MvMW~6xB0GZ4-Zd^4_#eFU9)RD1a?!RhkDzUoAS4!8IMNC`FdS6ZPf0z+"
            b"x2}^S4EWXy4$)bySyv!il|-Y>%2LvuP|Qp;(77wE?<}3eRRF6&>Ms=uU5F=Xqi`aRU-IEKvT+40RcUZqQz~|(-1X"
            b"9mzPybs^-w6by*d8Q~s8d3bqI=@48~M?Pz5m5IiD0x~g~83N<!gE{koKSIdHYZ}L^SYolddbxpk<j}8tFMx*Pd-b"
            b"7h;z3X;Ok!4Z2+13rlSJfS<rX7vgyG`ERQut+!*hA=U$NANg7s6c2uhs<xTIF57T<2|Dv>ZUZq7X&3+wf2Gs+*xl"
            b"`k}kumemdabeeC9)nd0rh&d_m^N-S>V^nZ^ALVWIzS^=TwdJNo+|2OjW>+D$7UQnTmqjKujn{B~o43V$l}0NfI9a"
            b"=_w*^BqMiSJbSHEsK^!IgHb?@@^t~f?G5K{EmrkFiNQz9Ywt;pVEknHaLHf!n+thHkyh1u=hHY2i^G1+67<57fu&"
            b"N<v=SCqH;?q*Rmo3bJ#rv#$-m@hkm<*KAIk+t`mP0=;?2;-WFXtpY@5f)nD8o}31wk(>iye<(1H%-3D7(sPIf!Jg"
            b"Aq3muoPzuo2YlPSSrEg%%C{IVpXmrNK{({MI8%>0){QNWSp=s0P_rIi3|MzG#B9hW`gA83da4%8+?b?{hOmakMK;"
            b"q08wT@Z>5Y^XY{P6Lh4t_%Om<$43{NnBD>HPFtMA{4T{LR_h)05drc6|23`J35dhCaL`AEt}N?1%G3_GbRWe1YD+"
            b"B5z+$7t=S{@!N}w+36y?Tue!c&!lgci`o3U>04aG=W3C6)5Y5h4E_aqe=>VFKb~bjzMa0AFMi67r>7_L6N>L2PPR"
            b";`GC13H%X(9o>^Z|~jKcg^%d|s*q`tv|gF7RQ>ITV5hd8(@+il&JL^lZ@%}=Mti}|~mVAAyXcy_*+oe1V!%oZ0vW"
            b"oO@fKjVO}r1z(1r?2OybBQ(d{qY%M8!B*injOzB7W3EhW7^)&C7{={>E-;JIp&FA75#!Zd!CX0x+FZZ_s2is{`hZ"
            b"8fBX+(_t!#c#b_A+R#cNkvqPL}*LByDzY|B7oNbAs5nV9fY<3-Hb%c5bsS7+UZ0mKoL=y+%eRtbSt1>xZfi94WS>"
            b"B@I-gn3*|5$Ezn~e7*9(ddKC_>stG)utHo)=$V(w-?R)XDjp>mxEp97WgbI>$h-#xI^eJ$y-f8;vc3$p!lAg;J>t"
            b"^m7H0tBkBfYLu#HPU&Z_Wc`F%$tT3DtI;%#C%7K>%khhX6U=mDDEP8&8d|}1k+&t*t=+w^5bCxMd9%ujra>Ck222"
            b"G{5ORlS52m5*6}(al-C#B%bH-ba5?0mChRjB*Y`JUhax%et%J9x$oIssGBq}2vq80fJDuFwPbaqyJTo!Gsm7!yEQ"
            b"!H=us%$rCcBC_L*4?6J-q!1tQORhOoy=ZO-@aL7XXjW&;4$tPYH{M!k>8aqn(*%9-l*vH<LmNcu{vyT^DQ^<B4iJ"
            b"s`5zfjcytzbMRVU<S9Cn}gR!Q1_o$!6$ZKV6W@y-76^#yqI;g=tD=b7CfGn70b!`J{DO-`PXn{laV2Ko4;iyr7c<"
            b"Mx^1(wdHSnY_!=-;>KOUZ_<1w?${f~I~}m5BPSDn53Z3jo=_H<Trq`)EuU4eeze`kbQ`Kq-39U0xn5AW%O%KqycL"
            b"T0pS%S71Hjf$MtX(O?ZTmoi9In0*B3E-R3xt8_R_ys%hd5Txy|iXh5XCFeg^Cyp)><}oML0jsKZGVV>$ENN>u8MU"
            b"$5;>=%<RFYcHe#+S3X7eCXn?$&m_j?uIzuExC^zs)>q9}4rgc6wUq2VV=LFPf;+!URC2$dZrK5y==KmyrbY<Hpw>"
            b";NYYPgl66!q=G1$8<HfM2a4q6OU={A5sH!*+aKzu%qd2yX(kYNK99>kk~a9dS2Fbv%*@ZX#4q5dWpLj>A3$K9kb5"
            b"}-|W_1i6@U{5NyEU+DjCN3*N91j~4WnZWSJbP|2J!lL@c$En>;%Q(<2921`_YL`zq^f~uoDYLHo09M&JOSaKg7HA"
            b"xVtWV~_DTPBIA-VnR7)H{NV(Tfa8ZAGCOQKOO4-XAHE=bT2xS>olE?nFU?A7vG`T-hQqV@#mFCVAzU<DT<YkdBh*"
            b"@T<O(BQ_}U$jAP-ao*<mef&CKw}smne^b}ni=suc4|ejRMDLhS?r@D@v5AcoPIOfuc|`qOt+v?FhCrI~jyh(^+`)"
            b"$Ez}|IPk@^B#`zn^!x~gK16*R=yNCFHs?5fh=5+{uQlC3;4tajIhInJ2(l67#_TkuVhFK-KD#9o(8E98Ou2RuMBA"
            b"FNCYVz(-J*(PuQWoSjQP%J(c&9ZC@Q0u6eY?I3^UHKikmW57fbx5kba=aEENa}XyE_!7tDZ3>#M`RH5`uXBfl$0n"
            b"VCRKak?0vqhugbiFB$+QdEn3jGF7+$R{}%)GQRUh`?@~Kjve}M;>_Ad>*J5_;HHHSNqUd+TuX>H9IWl7eX(m*Lx&"
            b"~hi&{0w$KFl-bGG7$yf*6YC=>Fu@`Q+94<kk7))%oPr`Q+94yLojoynwPSlK+zzI-eFYkFby-D}SWcnMJ^&W9;{l"
            b"D}+WC_@`)zLM5rZj##NN9B+=kj9yYuA`3ZcUPSU^W>|dV152q5l0vkUk7%C3U2%|Dt7Bq^Dsod2L<L7i>R;A(MPr"
            b"Gh+X(`u^iU>1MJ=4bBeX>JYAnzSI)F%gYCu7ueG>`_)l(?Z{Jgr4tD0tjjmwsbWlTX6oHN2w%f3Xm_4ox@;IcPmM"
            b"O%@$SVL-Nt>vtsb&v(LDIb^?0{-)hUvQ=L@0ZfwuYMU?AsXHVvRHYNL)6_i_3q}Diw3J)#Lru>UtB2)Cg-i`2_~o"
            b"~HNh-kNlP#RJvj-E0@i#4vy3|z!7h`a748MslMsLcjtqn#*(b~*qEG%pgP#jNeMWNMi;7eD`;<>fwt!Jk)m2s8<Z"
            b"K`?^-j%AiA%!O9_wZ==q8Y}14C@wbco?Crv`??xeXr*`7{km?nwjEb6{uo!NI|l<o)897Wd&$)Bdn;yBrLQaL#1C"
            b">(Jc1S{3M1jV!UUy6L@o%t79(46tj&*C$cz=@qQ&tSm@mM6T`%$71Qnp$LEuMuW%FO9>l_Oxp3ZeV;M&C$=ZIrO~"
            b"Rp-(oqhwqx=q?PiC)L_40y1kOB0ZOS)StDK^^-mTYhtXGvLDfwFE6{~PU{%4w_>WUjQs5vI87ku~#5XXi=5t`3lC"
            b"9(v$lA3xwN1av^Mn#N%$Nr4bG0*Qzl3ry(jxK@sg#>7LyC$5SF#x5xpPaC|-wOsdonIcf5E67Z7zhOT!az99AqK+"
            b"8Ru~9ixW-^Kx_=Djh3X`OIiP#We3<%)+6Afk%s^1UR0bu_<~?%*OD;44D4^I7j$OI<Xt*q97xS}=?D*{T-R$Dq*-"
            b"5YP<i2n@<Sc>$*ko=<%@_lL0sovuo^4J#5Zv_C`9#E+MSKN-S;I1VLLGO$;+E;osi%T`Nud}XNI)pZ0z7)|a!$9N"
            b"yQtN-=Po8Y_{b!QjOV(AC)9o>#WAJ~LO0A6=q7D|KX5prkS-Or5>C!Su#6XRu;VCY*2}8+Xc*odkCGjkfwMHw#2R"
            b"A7JXQ5EQHJd}?|vra{W5NgjxEF#II#+A(|$GfTe*!{`_sN5DdKp8rf#;6Lnv>Rmn<qxHD;{F(F`Uv2&yT_5<~g|E"
            b"i(vOWFTVcWU|{gSk9z*Iex(<S=3AIT1j+dETr&N8w+S=s^WKoz9f-Xx^e=KTrxWBTa-l4DbbuUV$%7K<B%dS_`@2"
            b"ju^Zy3=o^psL#>z{el1~w3W-+P3fb5V)-1DKMX2`yG)>2MWLuxiUu?;$F*JzbXzf}P^QLYX+hyt~)+Bmre4)sVrh"
            b"T|YEr?~LUtvE<8ZlJS*NiOODhVvbP-6I*_C>7hToT_+g(X?RUSaF2!A;8AUw1k72Tsg-5ZT5%r9O-R;Gawm<<&NC"
            b"@J{2kR9nwCz`#-{qr4PQ3VMh{)NxOvSD4cvz8;f6ONp$T?zX<ExdqE;Y=3RK7#v#~9m?aO(PY-weRb3mf~u<Uyso"
            b"%e$r#`+IJgzE%RvE``$q9gxVWnTD!u!aPPQTTYCBScjn5TvQAM7tcK8Bmx5XME79jy8h?eqTqLIdsdXg|wWlx_4F"
            b"}JCJ&(Ih`)(TC%1?Bh;r9(8zlIZtHe@EP^uf9lR(R}p<(m$fTs&?1v1KLme?Vn$ms+<e&tMLm3KjXNj{PkFhO}8)"
            b"C$n4-uNi}At8h3f`H@iwQjc>4llvTt+QZ|ieI3t!emTfmDi$x{N58kS-rTlSQ6TzEXwNpja!ckDgEgKSGMB<@u>P"
            b"w+eDi79*eC?yimnz-&KA819V!dTGR{NA>;S0XlRZ(3rcQ8-dq#^1E_dyyJAD8Rh3Of!(N-~^>Y>mfK$srbtG~>wI"
            b"D5xLP9I|YvaZAV<AvE2Tss!3_K@ma|m%P9y`fFWo%1*;MFw`U0A`LOaiR7#bS%=ZH;9!xZ$||ok{IK&^880*ocp@"
            b"iiNbU{Mxf7tnEiOT*f514eLY2jxtD<{hI-aF+Y*8Iv0?G*qFHN&W6nL8xo1O+SW(PPVF03TmWMb}8kepeb)Dvf6F"
            b"b^fX&5jI}0UhPcU746GAL>0(UK4ZDlk@Q`dYOWXie$n6{NfjF$`tyIilq8cUBKfd{)=DKS9q-t)CM|SfrK1uyF;1"
            b"@_>qF>)tm}Ffj~NtQ?Bf}K}s=FpXBbP?v9BTH?t+zutxTdBsv_f`OuHq4xwI`kx8_FJc3jkQhub|Sb8VmaMV;eb0"
            b"=me@r=JcU0jIMJKoGsXVZ)M|4y-`AO+KjmZ?ABI_!t<#1`WyjX*Jyo$6`tUVneL#o8ZPJp2r%A_gp|0HjDu=M_F%"
            b"rBD#|ug#$-Q<3(vizdB)6|uEK)`PqkZz>!fF+Bs2884}DY{{CW$1+LWFOc9tFtl%_^Ik6Xt}SSH?Z_l8xQ!F|$6q"
            b"if#NDdyVza*z2O{pYLU<evOY{*fhhp%IuDZM|w5CKp$KqL~ON-fHtS*n5ze##^;)6rkibJ4K^`2v*VuBsm38Z3__"
            b"OkZgYpe%=<D<-$X@^qDKZjNVqOWZ&Ph1m?99UM?4zhOfu@nD1i)@sQ7l<~HBV)vmop2rb3B)mKz^NpIsGdUADXs>"
            b")0Yq90S_d-upf81kjw8feJ*~9U!4A_CH;E2mV)|4(m7!|*6O|uumYUfFs#F!N?;dJb1>!zhtQKOe)nF%(iTS01R2"
            b"Ws%>|&m7aCPXYntSfbDpp8=LFw=3hMbW1P{VTGC2Xp(0t!Dh)_9=*n!u$B-o%*pnF$0AT*BJ*I#xVR@>H1^MKph*"
            b"OL6z$MjiSnN`{pL6xhC1_u4ragFJxz9MYAOw}7UD5{JC@S)!g!?x|5nc5s^&8S^$l6JSphNKpO?Sv}P6W1V93>_1"
            b";ea~s7k#IRWsE>fVG0oEHU2GW<4&;(-;mu?wB=&b3-5JoZAjDciJsbb@uGEt2^NhM%uiFRwEq)`xR18ha?(H<DlU"
            b")-7@qyd=^Fmb;0_aMqF2F3n@qhqpXm;5_eZXwc=w(_^5@UN8(eGZeG?%AUOq0^^1s7BpneSj`>8e@(*N;liSLS}c"
            b"^)()FTf=r4c4F}tS#!h^tC{Jhx{rh0Traj9UI2zXKs}vqqZ5gRPf;rN`9$PqhS~Uk@Xd{DJeTEAe+0^<Atr(D3ji"
            b"VkGAe|%w)X1ndAfR+=14z%MHUN~U33Wlj`{3ZDYF`O6!P;bsmR)VK!bq)l;gIsGZD4v*wE+iZR1cC<YpV5rSmYe+"
            b"@Sz=;i~X=M?atZ;<Pl2QmLDwedg8YBtd@c@xV7gL(D7gsG^MVV19G}uvT0_wx}c#%R$%-Fr*gZ>+Ols=`UWL&TfJ"
            b"F(+IoqX7+XVvDG4#?{A56VH;@FM#b#9&_U^DNI(K@yy_`Ka-Co8>Ot;1J9+H1<uYO4K`7kWr0jtHDNSHli(i5}aL"
            b"6Jkr)`N4%?B131!S*0M_diMhJi{_UB7x_iSlipDW<C6QEoZ3``?pv}uy;#k+h1J_jFAX#*ZH!DxzW#lCs1sXaicF"
            b"}G(}{ta8Pjo1oovH;4JZk{xUzC86;ZN-H5(xJ25yFVl#qHmb5*=2wo|pjP(89qs$YZ<+o#KF<Dtvh>)In?+d(+XV"
            b"DVyn-azXAslh<)a|x|KMykZg@1}AO-S~0z}X<F$E{&{OD4lkl(4%jgc)nD^A%68rBi5A#!h3{q{8aS*%KlD+0*CX"
            b";R-O@4Gm*DLYYXzh!)=#ZCmCQJ`?h)tk|m0Ln>d);R(0CzP3lYm@4|m=lwnyDKS(2p=lGS)3588f{cr8YXMXC6Pa"
            b"6WSiVqX4xYpr#KoyP*z?ve-$<5_DjM|lgv>++c>|Y%4r;i)_NP&^xOjWHptloEFMe`RQtj6ADE+hOIXyc)Kf9dcZ"
            b"T`Il5W*m=bv3HYszG-C?r58%W(i6@#MI%Zj6vD@-QEV!#oG^A$)3l|9)`xP0FiZ%uyy#zA-qL|pK;_-f{2JfRu1#"
            b"9{{b-$g#6utL1~<G(Pb_Er6lIo<ri2wc}UZ(y*jx4eDkDeg`(@}uAf7%w$xan%>jG0KY^Ee_JJf?;0$_MatOGlq8"
            b"?J3248H4WFbK8OWb1>VD*<1fY&%xOXcVugZJMRz;l@``@|=m`2Jo|N++p*qgZ7Coquh_(m9Jdv;_D2;@BLp1J_8!"
            b"OC#&Z*R{_!3t9b}M!OMmF>@g;6r@%SC$Hp0ao5Vt8Ug9e2K9pjg&h8XEWFgdZ~<NVaoG9_ij0{4e*TAx{vi(XeKn"
            b"}7^W5HD4-Lbt6c-tGXL0PeXk3UDk1}YNItmYTEdRMvL^6=j@NmbY%??9J3=4%k;-p}A2da*Dtn$c_40lDaJ5Woy%"
            b"*!>+&((~@C($1^kCK5d$d3=!3;W9+&7D6OnQ(>z9ygDlJJhTH2z&^3?gt}zV7r-%o=T!)VpqKM4BG9a4{+2Ea@M;"
            b"X_SVH2qx4ojILQGg{QIu;^EM<70%L$W$<z?cFe&KKvHNHV<vfpx)#@|Kc>)fLi#d|G*}LhRx6nR6HMFm8FqEI3Df"
            b"Br|_%{yP*ZH;`g$8kft91`+PR~y17D(x$bSHa;?CG+$vS$;tsUWara+md)OnAl%-VkPKS{faK6t)8>lMeHpMEi1L"
            b"ys#Gh+R}D->zA8WvF~&x%iM6fTPyf8PeW02`iAVq<*X~Kd)p}ukT;FSDd2Xn^S#Cxr(LJgNQDbSx?fYwlaS3Vbyl"
            b"k2&Qx~X!BnyRSl}^c4xXn@E-PnLIYq#+C(w)@DF>SgdIC{A3!|)c9nlutm%ZyOU_1qFC6W9uum%V!YuUxdQ6<UHv"
            b"asBF@c?z3@}CUgy`1@qSgx29&s<u@x2v4akF%5PczSv=KbbCOev_h4UUr!kGRYn^()Mo|cbL+)7#A@}<vu)Tf#`m"
            b"prrff`65cj4vhmJadm}76?!WePk&b=3!C#?FD$H6isPS6XIo|xQB-lrXPzR3S#;He&O-$_1A(jml=IQbA?0hjjJ)"
            b"UJ3)06qz%k1qrmS~&lH}vp9heEXAkW`OT#4N@Af+mq*9!1HHv~uWY<vcwz_OsvQ(C6LzSMZKDXs0#$nE%{@$32r%"
            b"(u|Ir&{H?be-D@{we1C1@Z}DF9?EiXx5yCl3Dl|q*3-ytA^YZka5`1S6wEryL~N`SoyE0czOCurdIL78hv5j?u0E"
            b"6;*T!}8jU-Ha^q|+6vJkGQvOy)`c|551feC8ws`u=BeF_eu8<-1&%IVW;SRqLz1<6mZEO|bJ_gbZ(r9>Lypz-Y}P"
            b"X{kU(*Zu1@xw|3F^kmUyKtqs23&=vICkNH_$2uDGrK&<8yM{gGWKaKd_!;mF@&eBj}CRZw`D!R$8(PC>vwK%`OK;"
            b"WP5@SHgT8+K%+9Al80lJKhmVf-b&D~lTeOBgaoE5#{a^tz#Kr<A_h^6sJnj;K?^AXzKuuOVyaSMK*N@HyN*N0#M-"
            b"lj~d_p!>+13-YZK0!(Y=3=ClR6f>%9_v)s2u@qhf@yb9b&8_-M18jj9Jy>9BOM_;!tgaE1nap2T(<JSSvJnmQkh{"
            b"rb?%$YD0&J4%sam9GxafqJ~nj_gr9P`C_)X_z8!0Ue8bG3tO00UD#Pc2#va?N}b?f=D7Y7+>SmlC7<Z%#Gkn73v4"
            b"xQxu(ifZ;?muq&>(cq#N@_8dNT*_XhXk>)G^j{>}W&eDPBd@+6Gdfjl96k>|-YxY!`J3}VgYVs=h<aa;yhko+A8K"
            b"l|qU8M)N)vB?eq^ME)$4{}cpQsvwRdawegwikHL1US=JS`?5+Ba_XJrJjha68*Sp{vrUcdZaYM3AS3kZ~<m3e9}1"
            b"R=zy8fpDY6fQ7nrd;lY@|?m#$ch#sq}pHc%Z<w91A{IK?0a{GX=ub=os+~nwI^be7|dWf|8uFSBP1TV-4ttdDLsS"
            b"m=7)6+v3+aHnM-Igs66voc6F|w~VfT8l$BzkG{D8;s9Zd~e>wUo%2|7sF+)p+u6TLHqi1Mm~cA3-eHt;cAv-4x9c"
            b"`;8CES9^h8;^&+tKs_?rV@?4{@~7=NIhGEJK^oT*JU+uQ)+6%2joFSANDxmrhx&GfvPIqpV#|JXP5Be&+-g0RE!0"
            b"HNX5BRMoN_fpCqn0{h0lK0^VR}qGAJfK(O;yY$p91JT^bTQp9J|$%+uK<Z$jz}hWC4g6fb6oA9bQVr`}8!TFmb-f"
            b")90rFF+a)kDb7;hlda*!{ZB{)4hB-RUx5;A%knc;bGzFUi%#W>!)wnMwcP6obW91KiB3MvJHUBQoo1@bRF~>8(ww"
            b"r${%P!4AVj8vTdTxC-!24cSDHfrhem=k8uP)R1(z{|G^0kU8Z^%DBbE6(5uqo%J0i`Y-Tx`y_+Lh{qgM-d$#)2K#"
            b"zNZ@y4~m#Ik~Sq$B&T7YphxAC~BMnRokil>H{N$LzO3(k&GLsY@TBoyL`GkK5X?!EbM<j9{<k`Ywlae@}wF5ai0W"
            b";B|qz-tVtO3mOYRK_vcw`(Q7A*)LuTBF<jkS^9~jc~N8Z@Wx)MckuQ$sTty|I(;i!zknO=$%Rhyb1#OCy{NXWsAi"
            b"Swwl~YMewJ~?PS&$-zV*@$+ywFjK7BpauN><S_)|YACv2w18Ipx@G$-EsCyq6U@y$QQ$0Iy(vW$+PEr2<{fK6@RA"
            b"2f1IXd0<<%s%-htdZe<!=*Hb-ItO3uExZIh0iWPdG<-Vq|w={Im_zjGY)r_h_e@Yhftm}))Vu5CAoUg;-;jxH@`?"
            b"D`*n<>yHfQ&SxO_jh!~$Iw_O*V*iW|t{G4vS(W~qEP;zZnW)+Z*PW89Con)Cl8MIF#($V7^c#a8O=jPxOQAubJ?>"
            b"jmB;rz{PvB#OJ)5T);!#Td~@`w2%v7SxN@8<*za{2ag_uba&HO!_Vz&go)9WeuhQ)#rt4=A3%4Z|GH*e?yPZx}T6"
            b"&A3fi6LS4tU|76$(U{m}#cSOU7Jk?c=Y+b=V3%zRV}R{TkI|@itH)0cS&F7)m<A(cAhL46r=YuYd9_vu^DbRhMA`"
            b"F?2HuDqzCo{im(Rn!!GkZW$mru9%;E;mz@(&Quh&uZJ^t_xzdzO^n!v5IN6{I7c_D71CBS%+!EpFGdyh(>1)j1>="
            b"!5~6IinwO*4@eM_4MtVMRs<MQ-3bg5iuR0LbMWUN*o*<e6uUpt7umtQ9)B<QLVPP#b^LE<sI>2wGkQ=k#t~H8(lZ"
            b"`Cdx^hBHE@g!bG(mHJ^y*oKnZHIc8dsp7k?e&%bq#zr@lpDC*Lk@yZE9LA&vLI_>wYQkedxtk%1gUZbNI>qAe4(b"
            b"p2^?6<4RbXo6Ibs`c_&muQP<UUn{1L}r)8#$JXPfb8zk1_@F-=Ydq-&zqq>?T#cUx79o4CyaRR-eHuu17|OhJ4rc"
            b"@S(8BCD<mF#ueCJIuiX#bBc86mDtR*vNa_(Nj0y=<}iU*Tf6%#v7(eNxeg>(TzE}iyRW-8ddG76H8kc8q1fXZY*&"
            b"-h08OK4h{w4SSQe>0X9@4+z>;^=TY-9I@JLO?@t2m4#xSUq?hyuPhrs2h=N6qQT^B546}Jj2W7b*3f4{H<odUHfD"
            b"&O2y2p_HBRYJXQLBAIlkXIeps1KfKk|E`syElha3+>o+mp5;)g?Zw;zt&TW=l0qs`V0yk%kcy-HymK$w}(1uw_L>"
            b"xa>;%XGYD?Ih8cvGCjt+srpSF%+b(Q&^2KspGgt$7S+hL0g<S<0szzt2P9t4#h9n0JI)YekV{A&Ic??kqb`Me@`@"
            b"(-3aT3O>JugVxpJ(Bugd7z5*U{lifeAkz8tCoExW-l>d+^QQ0@EYi`z`Q2(%rcN-~JcjT9wbzo_hZp!yWfvy$DHW"
            b"1$SocYm~_DRp<&!vx|We`X!cNZ%W=Y;2KM?!}=nQi!8ye=}YdB1-7qpmzXJCWNy|LpvFN=P-<O}_Ez{f_$=s$>hA"
            b"=sJ}mL=K?)nL*q@;IK<%8U-e)#IWwW9O5GKe(GfCvNmm{VV7SHkO0Z821iQUhZnx3Hvw6c&IDs9-YILQES6IUS<="
            b"duN=MXX6C?j?&FB;vh10i1#(Z4;4kkv43N8|92!yP+A`5^s|`GAQUK7w|^j6yUJ%n_MI)1}7JQ!*DW$FBT`m+T(I"
            b"i>_{%TZvaW*S3nfWq79IJF%*+wLQIS|J@Br@oucWuKMXc&=@VfjX@xW;2B}0IN|W&$XK4HLB<12h=)@OLRd}&~RM"
            b"mmT46X98VY93LMiE1vX(G)Hl4tyJf`lR47LOuj<LQs|`#JxtEZD(~j3Xv?p%KZ8T@`h=MYJw%^5uQR?~|h8MjVUe"
            b"LCQlG=cPOSwt08khan{az$Z+G^(;*V@<tw=v6xYxWMcGRsu%~-m8I68ywMwzuwAB#wK6;${_xgI{P(mPkG_nY)|C"
            b"laI&g~I0WEl7jvW5QyM5XTsO&CxLF&sw7wmUbvARI|6aW_#8_zBfKN6bh#36<V2rm;jqy~bM5s1npqyGbI1NV;"
        ),
    ),
    "src/simsopt_jax/solve/fullspace_native_equivalent_quality.py": (
        "abf9726e487eb4bda9f82c6092415e988e5a346383c89cec732fe7185b6e6fac",
        (
            b"c-q}P{d3&Lt-tH9;L{J8yOej4oY=3UmYJv6=TxaHJJ!WBt;a)Cca%hnN1jQZ9d$bW+aLD(3%kpcoc6tDruB)$?qa"
            b"b37QkY`!^6Y##m}=J&zG-{lVV+Lhq6nq+Adj_pQ=?k8>&sYPMTt<K9#fb<F5Kt)MYayA9qDv4R=Xj7TxM*esHk7D"
            b"U(&Z*;X~c*KNB^+Rsg&v`u}Nv|H$=`aAUex$SO~qFE;cfOqX~C}(xMDr(tZT@0kVvu0gxOZYz+3jW!)75rG29Slz"
            b"}>1iK;Apoq(r07b5Gr{!oj&|Lb+oCIm@?hKc1DQnCTqjpm1A_n>e*jFIp=x%}S=-RQR$bfovl3<p4`KSnZe0z1(v"
            b"_Q{YLfC-APY2}9~>SY9vobC?Iy|dtKG2c$~;f1&9?0Z=%;B1+GWpSR&8CEEBa(!T&@J*)1t16%erJh>tZNYb<y`Q"
            b"Ee;?brGRu<ulYalXIF0PV#QmQ&2A%_oHfG&{$wwPyDgb8fAyTqJ5A0DAkcERg>et~n-9gW+Rr((-`*ufpL}Sx;<<"
            b"bVk67n@wdr9+bJF6vZ0B-y`}wwOKfrv;b$(s!`aW;U&qLefL$~XPyeqE>wRyjjg9QG2B^vxl;Lj=e>1tI18?G;am"
            b"SyS%djUN5y6kuL;D>p+rL^}0=p6|5$VbeAwdi)u7~q_ULwYcE$=6<zwuGVQJuFyV=KWAym%v$d-EU!?O=>-d#~0-"
            b"Bg7K+DM81GI0$**|Q_js=cYLBCWJB4g$2o{YKXkz91E&@CD*ptyww(<i=K!<qt2M~62asS><{)hObyuwKM-MqhZ&"
            b"K!^X!x4He_b-fcPmAuRoM;I)m<LKy+`qag(iw{UjACFh8#o!G|?(2+Gik2i=$$`9l8?GzbPI)e!>N39$VH6(1b8;"
            b"hr&X@U@_Wm4Qx7;&N^*x?)nPahhTcve@?K;FWYt?E2}tD;TIR*r_V2Px@qJuUoFpGo?kd!^dGn8G@oAe7azCB3$D"
            b"(eRmF7!+PPYBJXcpeX>^<mg`cmzARp*^#!&|IP7e+)p8jd^m(%>s;`Id}cEZ$Ok|l?XrDl|+W)jg^Q+}LXH$!(k>"
            b"!>!EeL6loc>VI1<w9#KH_5{T_{pb!u{>Qa@{6;-X^{VY{6vFT{QS%5i?iilEucq_A8SA_fB0$fbb0n>k-uJCoIQg"
            b"UHu&-J(ZTsC1yR7u)7L*PmI~^-`FFp6^ytxdPac2!?eU{0$B!PDv!mbZ<FJ)dE%;`_m&aaFM)FPZYqi;J@`7t+kj"
            b"Qe|XTa6gdr&Ex?a*i69v#tTIG`I?^5@%OS%OL`hwhGO^j$yUTh;W0b^tr->8GNC4Y6Vt1yTFd1gRRbsZ%va_OGUG"
            b"X4mi+Y;1r2^D-GkuXDO5ldewRoSwZn{o%zTe|~m;wp_pzkE|ExFVE$>WBTrwbNQB#S|a{TS{uu*XnHUt%BNI;fRq"
            b"j8MD3<(9?wcJ6TsA{2+Jl!+RF7z$z!58H5o*oTr1H|)l9ZswE<aPx5EkDnNPc}xT9}mL+m9eq;^BbYCHGU6{tG<D"
            b"J3@9!!AqvRmn{<{OyNNTVM&YH1?ZMAAE1X>d9e~tp=wzPJ?o-zrZ*MCHuKz3~4J84x#FyPd-<}4cKO&p0{<is=!X"
            b"O*=^S=%Co;6I?=TxyAkEaj?e|zlT79=Z-7fZ4X(<fuh>X_crnp)%54j2WD8BX;C3!LdEHX(;r)!skF<?p_13!;Sj"
            b"NQ_G41=iW_8oG4QwZ1gj2g%<2td2&^{c7kVeM|W{FT3wV{$dz5=A|Ub^CQ&RMG}eRTE%NE@RwHu#g61z^dJCUhqe"
            b"NvADuZ--neoD+j3t$7Qn0lHLcuS^A@5Z*!I1c~vtl2XLi>4;^5f+?pe0_c}s>cQpIW6HP0k6Dl5?w=V6m<5iq_LB"
            b"OZE3g==HR0)Pz3UTZiD}Ls*(rp72EdzbQC&COf78Tzbpx&(Sf*W1*##JbQHl7V>Q7>+=gbT^iFh4s-KA+3YNR!g+"
            b"bCwQIV4wmE>Y8)qQ7-GitC7zsP<s<Bkr~AkQ%IxE*A8LDHUt5mfJ2D^ks{@$Ljv2=&q}#N0;{k)`=(~x@Br1fHiE"
            b"N1t8r2fo%Crc}=?(+I&-%19kDQjSW)?;D!$A%&G-bv?0R6-aE7=+cvRdY`{2N1<Lo=5i*STr<^8g%tnBu$m4KI!Q"
            b"t-}m3BSxt2&~&{iF&2A@d7B<GRDgJ4?<H%0mb!iPJT3B;k*yo!!9ZF4;*ah%~9REwa6$%)_EvJlZ6g=(6ZbDab*6"
            b"rM5CDe}4Cd0t<~V*)J?HasaaP;5LbvDHrC;cGs-EFk<)~M(tfo>wjd#X##%S-^(Ujb~_+lUxSlKe^0}L{;CDr>P}"
            b"fDo6T+@Aub*|RE8M|ws99(lX<cAsI_42^02vp0&%BXDRnJ`Q8i=-V4iU4myQ7zp@cQPs@q}!ufXGbX(n_Jo4`pzC"
            b"J4}ogJ^F5IJDs`S{O+UfH(EHkd{0bmwj8qewPnzU3Nsrswp1NkILEcBdbMM){Mm77DqmH5NToQJ?5Uo5Oeiltlg3"
            b"AG>wAaQf}562>@sac0#mR_PnTjkO<*eC3#Nb%sF|N=aasyucpcD`ve@5l8FSDX6U8a&WpYvJV)W?C<{+zBT87LnU"
            b"AMc%jznjkcvN(|H64wp~%pzz$<-2R%Ow3Z8y;l2s&~C0E@svUG*J_c4*Ok&6GKG5iz*m#KjDg+#j7sj~^$BiU<xK"
            b"fiki2#bX|T*YWX@5@yN97KXqBb6~kGMQ7sDES<(sK$$7o?D`?ODPUmM`vl08PA%#X=6T@gKAs$f0Ga%HxMRZux6|"
            b"Jf_(A3K$#>HP?0kl05DY_d2`~nQG$vuEKPoVzVSHI8&tE<H&cw562P%wpO-3bqDKTp%fd2F$%L8Mvo<@zvM4gg^y"
            b"lw6#Tu)@l(fsIv3nfuudv)7@XtyX@3LjIp*!CGk!*1ZGgjCnPsS1q&%)hAeWSY{y{iU%c?*8X#650SgO(HvBtd(~"
            b"Q^fdJEfo6lL`6m^yObM8JJBCr!!WrDEt>+&{ajdWthZ~sxO<S*f53dS$QDYuZ!GvBnMSla_<2|GqwH=FnBxHCJ&7"
            b"_TaFQp*5TDL$VLfZ}TR1$WumR#PMJ1jR>w#6N+(HbUcih_CY;hg2qIB{pJ8*=<%vVeJ@V!os2B<3SOfyTn|*(2aS"
            b"&4qy4+64horV;Ye-26iQ8>R$`^jHnFr}o)X^X#v6?xk_|(l!I}=3UJ_Y3a#*PHk>rk`4Hxu!hAjbQ9-5D@`2e>D1"
            b"|x#0hmc@S%8Ch1=uQrxDc)w*2!v?-^QP_d8TDr+R336_$+o)siXP1}+B&`Kyj(x|C}YmjuBVnI&)EO4s&X8VO{+Q"
            b"@uRrDG+*3biB8bE*MsHG2aIAETTb5=tx87MY+B%yT26vnW^8ZmB=5F0h?`I+_^K7P4!;V<j_VD?e0)VQ^c#pWbQw"
            b"f-IE{-8O|3iL-C6q73q0VVsq^nrCQLdG)N}G&-_TKrmKO3T>CG4#xo-9t$CWBLkXpx@%Pq_g9ACJMh8RVYi3M^2I"
            b"_i0B%;v0u)(k%OKR%+K}P?KLwsLf#sL61E!Zb!JL=5xWmC>zk~Iy9kds<syctniLC=}kd}KIR@!wwIS8tY*8)}!w"
            b"hCnCTY$TZX*t6ITH*Hr9Y(MvQ>jCq2%QkV&G58U`+_i&Hz6kyJz_8GU=jBJ68MMXFc|c}q)^{p(l(2-#_>>J{gDh"
            b"7pq^vj~Ckc3p1wsm$1@GW}$AWh$9T9+?SRVALc5-tUM-#C$dZlLAZk=yn)2p_11%|2vMk9oNu2pUWNi3=QEp-4yK"
            b"nXMnQ{ghF)kio*EylLPu4}kmM~ElH)G$T@31|KZYKF;N7fpk*$bi2!iN|UxoZz~KlDf#uJs^f8Sr3duZJHY#JHhF"
            b"U%xrpJNwkcEeFRbN{%qg3tpu+;ayU&KeFhnmNsUfYIOQ`#Ncp0U5>>g=jc2H^*%9C<bbbaq@hNlHTTqgmVE0&F4m"
            b"gR_2D8$0P-fztnF1W1eTu`XX`4RAy%_IDl29pm?5SWXmb*c(2M~$E_)fP>X}W=)6-Tq1Pn>*bX*WTm6DOZmJJLA`"
            b"pI=f)vI|yVAB3}^1XPT_WSU4z4t)bj>PmtLc{K6iu3dBoqwU3hptaYWx^aDMifzt>UdKZ7#rE1vbH>edR_bUh>`x"
            b"lP^P<>+Y_OXogMQ0I9O6T>Kd?7o#em$<iziB;Xm;9MiO`bq7HHu<`FvA0>@EtGE5d$v#Q9FDK4Ga$$<WG{Cs%;U$"
            b"$PT4mvy_k%}Fv+&GKa4&y(d%Me<X?f?Fp|NhG*xKDD>x97ybiG8n<ugrlZSHX!Mefb$^lY1EfD8dy$)Pgm}d^c2v"
            b"7Mt+2;{dSE^+oBT9bsk?XYo$nv#{2ggPVe9Ai+5A1dpzQnj<AZ1g;aTCBmuPD4GF)hA+iCfV0I!2dQ<Y`1Xx2@DU"
            b"#Vw(vG3mT|@mKn$19x+!Z^JwLW=o@%VesdImGUK&aS4E7>zOx<-$ASi57fDl1DZRfeJ%SI~5#4Wv4d8TnmH&alTL"
            b"^EN&Snolml-YKqhQ`MZv_6KsFK(s7#(I+~dsWF^KRP@fi^5G}8#kYR^-&sAm5zT+<jhW`ky&kvI`X5M-Hbn!jKH4"
            b"_R#w>`-P3B+fS3GghS-*SJ<SUFlEC|62b2mQ~!JezzaF(j&0_iEGS?H96W|?NlS&2q~O#7XB+%XR6$TM!xVTvXTz"
            b"KvRjSwuvvpf+EDO#HAW?=E46JTn``xh#u=m<BG{D($(H<<RH+aLO>TO~8|}HyB5mpXmYJS*7i$k{q{NzP&LE+_*Z"
            b"s1?!AvswX>g&__7As2a_K<EgV&<V+<XWY(m#Q?@?&hNJF1dc&fdhM-d=ecAJajMxe5@6>qyQ?n4XduWz?y$24{15"
            b"F!Plc?*-2mN&0A{)NPsHuIG1Y3(>pWeU=l>9LhQ+M(%tN~+aYpXpPS^RoS2y?uJHjU6ZW^_u{O8Vdl3s|581B-hF"
            b"BQVXd)-QGG*J07yB$I>jQ7o@Jn}+>a_WEmn=H|_0<k}>By9GNOBHlQ=Dx3WD=G8ZEetHFVG6^9w`<xu4zJLGh>yM"
            b"MaAAtZ~e7&23i7u53p&4IGjk6|FDQjfFw4HtriMX1<3$CiyRFKQr(alXeaItwJMo7iVm^RPXVu-LR)M8X@FR1`Um"
            b"bt{t)uOr9!k@>H*yo()C;m={V<|Z}G<}z9noPUl$1#9SVq##10qTE!Lz~saL%5!k%WD(u?bOU{9zEF@Egw2>^ZFv"
            b"kXVvzTqxoZ4N*iHMBkO@n2@`+^P*UxBBpDev93D*U(6w{xUS>YHAv^8dU#Gr!E(hKMe_s}>TR!Zk58DYcS`u)F)c"
            b"sB9Lv&})^X<_)>z*p7@F)0b0qfM5x_4OhhWgAjM5gxTR^Eo+J+^jfo*PJhUy&=slK{s|=R?)pX~d1JFE=H02zzqO"
            b"ED1>r(up=#eL>k-oe(BszbJaoOfpV^IW-DdhkC9-X78Xng3;~PqDmBAqfr!ABZb!*QDpFfn8fixX%q{<kVQbO!^j"
            b"PBwVg?9GxbxkoPlX({H5tWg=nVz)&U5X8T&_5vuas0UAk0g$5_hD5?#eJas@N>Q@vJ;mZ_iG(TZfmO#ReOdBpW^?"
            b"JY4OIs1c!rYynKFwW??1F7j@T+p5DeK*rEsrdbH<APA~kL0p#iA6#WKbUEyUjvqg)S_9n>(VS<z>X00Yv&>+xhZa"
            b"GrJf-v68&6A*;5_?mh5RybMCpT?&WV^2La<A6<ntu{Qz_S);j*vsb$gZgoW|>rtC_51CYyTsduZ1qOlohAP2N=@r"
            b"t=!V`92zD_MX-&*W1-J?*t36+!}Fp7ibmrD8LD`^eBuM$rUA8|?aw{o@>^`>Pvd9=gTIjftU;KT~hdNAsi9G5$hv"
            b"UB0)Y1dbJE_${Vt4fZdXsnCx+(4msObL$@JWY1gV`$x0F_gsx5;vMKYEy&fg3*z!VI6a=Bw~P~d%NQPzBbz%F>6J"
            b">;&bIr1ACPkHuP2#1$(ig{QUvSKbA?^{Y}WuVaR&4$9N)!x13})$ThGw%Jc*CW&Yw6lb8<8JCSg|sNW6hEpqK1T6"
            b"*rlGF01Ps61W=Lp{T>R`IAxMD;UUq?Z5qRFe~pqlD03wwJCW@rOc6>LBqDiOAQ2(^4<*C2QNHPQA5qHZh)cWzjvU"
            b"cfQIho&VXpS1l;iOvDo+IdQ<8gr26?y1>F4?SZp_!>?wTPh!0?F8B>472Tk^3=98m{tXb<y*Qc3a-gY-`2PArz*V"
            b"Sz~abOcp2<NiKbVR+1;F<;{+nOn&!oC^P&#1L^J3!Vl;PWe6+ftEH<J{d0yxxRwJMcE}>Q-+HRB6VcErm8Bau5MT"
            b")q@ur6|#M(;xsc1{zw5H>)w)D`^<h|CSMw>o;gnxuy5Tn--}RFY7}Ii=RsSZ59)oBYR3Zz7iH*Eq>;r;nePP_=HX"
            b"532Pv{f+L~wRQ*5>LM(EQZ1@+e9*Q4S)!V|gNRrNY|{SjHAMy0Gz_AJJe?sXmH;U0{yK9TfgeyGg*EIhq+H}LxnB"
            b"B5NE@<U7A9+Kur&P&!TYNxyzW%9@UIgM;{nRdaaJ2k2bjg1WJgq<U~>EjM(^=<!Qt&PBekURf|<g^zRmGDr9p<0d"
            b"lOSq9>QZpLJ351XK4juVCGBcA4(AIe(lJXT!=QEfk+AE$Bh$YTV<F)auX#`>4VmbX{hHS+0*Db5sO|q6;8DQs@=z"
            b"uEOi5^i3<IGwE^4}jbZ2YM?ek{$=PH{ABV&4+%llC8h)#YjgAIV%vUw)*W8~IOPr+eiC{v(<A`U#{(k`zc2&nfCW"
            b"V3pSHw@BlndJkWz-xQ&Ha-EJ}i2d&~jcD3nXut=OaKDsDUGF?_;OJoY!t!3I$PL!KG_yxy>OBxJcJ_bUz@eLvQE9"
            b"{PPd<zQ(oaa^%jJy{Jro=AhmmoD4sXVQzRW_zM|C2{VWscfu{Tsdwl`?#bF+M|+#-D^6mOlF_4llh7Y6H%w+Gtz<"
            b"JW;kcx-x113PpaXye3mU08Gt&Dz)TaEQ02EN0GT+VAuLW428HLxEV%Sms1YOphWn|4f5Sm0=H&CJwu1{#T}pzGYY"
            b"a@}~T#+;Z$=9Fs4Qz6v|<W?Ppd@8=5W$fa+7wQ#RO1pIC)^}xwBZc5!5v+-bkX2e-%-aAWw-Sx7VzHy7sNPu<7H~"
            b"Bwx!(HcIa!wce^2!%24WRLj+eq?;xd)g#dpsxjKKOcu_Errsu<$~esDvyOhbDR&Gg@hceA|R(vdZL1bKc>vGVKAT"
            b"Fv(LD|3vakt-C#&%XN|POEP0<#i$}vuhUVqi;_w1Mf7tw7V;A>HdFKB;t_H`(Nn4=jn;5+Zo&CJku|Yq?WU!xU)r"
            b"IrFFDpeT{>3DCgwm+Qi*7ZcBvokc@|-sdRusiHBI9c*pPE<{DP<-v1I03Epi^I4S)=99#{XHh_2@l$0Q(4Kmlg%F"
            b"{!V=7$1lFLl+D`;78X?$HPH1D(bVS47#(|=8_)xFTQ?b%>tRG=LFXAR}hyVlppC;Ra2z7cKV4+<d`92bPkW~6Ra^"
            b"K$CfWVGNKrQ@Hk&i$6pfm7C-YYzQ=zEZpqomJ>9cxndkYa#**?`RLhu=7knWz-{v`{vBG6PDM#2ML!Oz9aNfx9#}"
            b"@kGq<4NtOw~~`Bg*8?P&G^(M?x{<yly|2-6ZHpr2)@L>xq|CnIL%u<GiVfQP?U|lfryU8IBYF=ofY^N1pBCUZ^O1"
            b"<8eS^MWSpEbVLCz;eb}-%5#$Im-1#e#iIWY&3jc<gy_s2zQKHI&+`42^}Ps&@>9EA`+0-|isDDdW+imBda?8Q{M`"
            b"w@L8NYdQr(&Nd?VGhr$+e5UcVi6pk1@csW{+bKCaL}Akogv751eFtCShLimb?i_TNsyul8dVXL(<)1s5Vs1P|{$_"
            b"fWN3UuQA$AXROlD`%QmwVN?@3IgonFP2N(c$8qulJtA0(VPE$0H;t{UdPZ6Er$abP5ql9O%SlXiz&F1RJBC1@g-6"
            b"8|Dxqk@H$4Qd}SjQlfiJA$Cy;CB{gB@tPtJ=zr}+j$GOcBH+!p@A0B0i6)rF}zv~kRKD2-`qYL6{!_<#~7Xp>ebh"
            b"L1UhO~|?#wiT%(5N6|yB?Pad&Lg$@6M>I=;^`O1&S<AY+=HSGGal-Ek<-9#xBD61we>D<q_u^{<kT%6W#Jk9Yaua"
            b"-P_N2af*8+cBxz>o;`;2K~%Ip=J$zqsEHjZTb+KTChs(Ja-qbGn!Ab9^;9c58OVh4-R<y>iCpjKu8H;KV&LW}2Ch"
            b"K5bWg`eV(^b4cmy;XC#%_Bq8r;(u%c?wL>^ZIWcw<G>>gSodw}l9?yXd^2Pm5COX#3%yn=EM+dXIOBV;fOyi3Q)W"
            b"Hv%t$HkpIF87G&6H~ltC}OgA<jEgJndn8Zk+@T?MOD=CwpoRK|3VS}lK2!Fi@?wHeaYJ6vO5p%bIKm5T`#L}sq8X"
            b"RN=a<gvg7dirV}!_s*7PLn;a|Q)lalcHoc%7{uvj?iyT|rD^C;jK=*z}xYl<BGEsId51p6y;XY+T6?Au>(sd&OLQ"
            b"Wva2>?l7GmBo!)$cCe-qLIoyrzrI%hUhJ!jo#fg({~BZgwomo3}YN!ZH3^#*wg$RSpfq#R#>8f>Z>f{fjSpHfbA!"
            b"SU}4lhf|cxG1ZJCZT%w0h`5S|xS5^k4J!K1HRbW1_$<2YXM*Lskmc2rH7h*<)q=zGP~#N>G#W%o5lp>TJ;vYncL~"
            b"2ose!3cKuwhBa%m}b(p=V`vUaW9ILb$v9#rt^b~5IZhA0cvWpU;CRv~#sJ}pBmGWtEh-B8u+7rX+WMN<Od;@s1{i"
            b"@}*YjB&RPrOp>B&pIkH{Z)eG{*LE$wd&SuHiBC#+@O|6J?r#Ab*2X3Q)=TTh$;pR!pra)3_5X-L+wv#FnJ{NW#}n"
            b">$<$Bms<S-R51r$W^vw72%02Z+GV@RDBDZEdsoUHN(6C=J_$!U^QRaD$l<PCw;g5s&yjervi`329Vhzkc5h|K&mw"
            b"BG+g$q3#n|a@iE-dIubLKwsC&6Enj~I>0NVsg<ff(0Y=VL+XxNi#W?-|BEwxn0mdi40oO!(p7e0y{>Bc9Y|J0$0e"
            b"pJ(*Tx2&SPN+jSWz2VyMXF2%@N-k$SvF2|1BigsYv0FS^<S*ucX>mZURpiiET$R>y-|a7Ovo5rgAto;N$G2BS(>A"
            b"m$HOG{Hfjjs$9y3uKgP3R+z%eHAxLK0A!QtoqO|dP>7s&;ov~OMqiNd>8zMvk21kz}P;q+Tqo3bIF!Z;j>8;$c4+"
            b"&-w!%iAa|0fb|u@hYi*0E#7f{_4qhI$nob(nvJ@VO(Ygo|U1#;(KC;r1#6e7;(fo_P&IiITkaBO*U4F{oio;g41)"
            b"%-2rbz#B`^aFtGC@VvMi{NNO!Of{|}*aRxWLcBRetk@%qKspJXy#Yuka#L`H9!L#tXx1=GcY<B&ST$WV!ba|s2h)"
            b"fLF`!gv%2IO2YCK<tmQAJOTuhG%3s0CzT?1$+%UZl&dXp&!(HefoCMmboHJ&6ik6JLoAYaV*x>LHd4>^~cJYmq{)"
            b"ne%P?;Bl0y{>;xFGh<m%bt69=_P`IyeSxpXpLjocM!#CfzlJwry?%?O3Vz=yvQ*lq*!WywLSxQ0eDt`_7K3H4IO-"
            b"bt)*MGjag6qnyFlh<L?Z}uQJJ4bv$|=!x94pG!hiBE`Ca0p0so|}3i^L4x{4edF{4p-VY9pV^duE$@nr{cTE-R2l"
            b"%I-)HxKL3i}HnMYZ4ug+K(i5SL%zO0z>ggV5kOl(4n+35V|10pOHP}g1W<stP_W4#eJpRszOA*<dHZws8yPqZ%uY"
            b">N;;`cp;KJ3kerc*w>>BKMDa)E+Gzri`HnU+M;O>tGFu*S8$OGf_QBQ7f^W_5$3*mq{E%p7u(Adq7$tMKNdr>U&-"
            b"C?Co^(!wgSm{I9iE>q&)zKZmp}ZpcuIdQPhbDISRSVQwM2RG^Dn0_&X#}8FV6n9a9^tqUN0`ro&l^6kWJlvaj`sI"
            b"E{sO>yL{pg^D}<psa}^0d&X*Y$-%~CrTBpDP0a^q7I=(6vB*ILh;6ezAYTQA7`$&jqmK6xU@E!D^T6#C0X@Qq6rg"
            b"*QD-=!eb4&4dzq44;Dq8PP0FJ$1gTY_x6dhdz0vb#Y{uiGlf!P"
        ),
    ),
}


def current_process_start_ticks() -> int:
    """Read the current process identity used by supervisor-authority fixtures."""

    fields = (
        Path(f"/proc/{os.getpid()}/stat")
        .read_text(encoding="utf-8")
        .rsplit(")", 1)[1]
        .split()
    )
    return int(fields[19])


def artifact_ref_payload(reference: ArtifactRef) -> dict[str, object]:
    return {
        "relative_path": reference.relative_path,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
        "schema_version": reference.schema_version,
    }


def frozen_numerical_source_bytes(
    repository: Path,
    relative_path: str,
    expected_sha256: str,
) -> bytes:
    """Load exact DIAG2 bytes, using its retained archive for later-drifted files."""

    archived = _DIAG2_ARCHIVED_FROZEN_SOURCE.get(relative_path)
    if archived is None:
        data = (repository / relative_path).read_bytes()
    else:
        archived_sha256, compressed = archived
        if archived_sha256 != expected_sha256:
            raise ValueError(f"DIAG2 frozen fixture identity differs: {relative_path}")
        data = zlib.decompress(base64.b85decode(compressed))
    observed_sha256 = hashlib.sha256(data).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(f"DIAG2 frozen fixture bytes differ: {relative_path}")
    return data


def write_artifact_ref(
    root: Path, relative_path: str, schema_version: str, data: bytes
) -> ArtifactRef:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return ArtifactRef(
        relative_path,
        hashlib.sha256(data).hexdigest(),
        len(data),
        schema_version,
    )


def write_json_artifact_ref(
    root: Path,
    relative_path: str,
    schema_version: str,
    payload: dict[str, object],
) -> ArtifactRef:
    return write_artifact_ref(
        root, relative_path, schema_version, canonical_json_bytes(payload)
    )


def rewrite_json_artifact_ref(
    root: Path,
    references: dict[str, ArtifactRef | None],
    name: str,
    mutate: Callable[[dict[str, object]], None],
) -> ArtifactRef:
    reference = references[name]
    if reference is None:
        raise ValueError(f"fixture reference is absent: {name}")
    path = root / reference.relative_path
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError(f"fixture JSON is not an object: {name}")
    mutate(payload)
    data = canonical_json_bytes(payload)
    path.write_bytes(data)
    rewritten = ArtifactRef(
        reference.relative_path,
        hashlib.sha256(data).hexdigest(),
        len(data),
        reference.schema_version,
    )
    references[name] = rewritten
    return rewritten


def seal_tree(root: Path) -> None:
    """Apply the exact immutable fixture modes after deep writable validation."""

    for path in root.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(root, 0o555)
