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
    "src/simsopt_jax/objectives/single_stage_fullspace.py": (
        "ca3a09f57fcabe4e448b9c50256bf28cc3750005cf52199ace2061d3e55f19fd",
        (
            b"c-qB1TT|Oe*6;omT0eXx6WOtiu`yMeS|ARaSq8GeOwDYHRkfuSC?iW=NhUz;kN-"
            b"|zq`s&nY(jSPkXU^^eeQkwbf??xe$3KmoLKAor#$j=%TJRcLw;d}X=WAo*t$5q{@u!><Sxejyg+x@3O8}w&)3Mu);d"
            b"d<F<uQiotyh8w^nJeiLsS_OL8ks;wLLg9#9scq$pcjIbOjh%rvmBHt=JGt!0`PIZV2?9#Fi&xrLITldhpIdd8WxPUF"
            b"abvf}6g!*~=X6rgnhRnXAN)A#{?#^?!Wxs~C3jr{`J#!rJzx7+P>!Yo}`o)>P4O@=+sidO41E1++Z7O05QB<F1cR3J"
            b"Y_c@7|yjq)J!3rAG3mN?n0$|l!Ewt!DoQ9P{yQ)Sh81VEzLu`bXG2REB_j2-"
            b"LdX^oHL2*Iqp*;n)^{T#qx>nE7m`kJgcWO=m80U;i#_R|ax{7v?Ny*r$)aFIQg^RdzxCwIlYM>={084W{(5srhhr`@"
            b"sOkDo@dm*awVV*><f=;24o5r8RL;Ye-aGk>AsU6yXvIPjKH3IikHhr&A%&^eu61AzbDpghW-R>~3?pb5-"
            b"Ct36&H`5sU{3N|SA))_Dk;efn(YznwK<|!=b8W$hZa8WJex?05<TLo_<%C9$B2!inYAOF0;NuFjO(@hrP?7DSjIqO7"
            b"lrU6mb+@aM9c?C-"
            b"Da8d{i=P!>LB?Z0%rfV2@g}$blm!Tlq<c?DFkj9%8HckM1HUJ?7VXUyMj;>XDxsPI$F|Z#%9MjA(wv+2}9K)BVH(8<"
            b"lPXbiu`t)y$Uyi-s7FXAwE-x(C>dH!~Pm~hXNqw%#`VXUSXYu>-FQ3mB-lq?>4s4?S1M)YUmDD})^GDkE_}81iU0!{"
            b"<IllSy+rs;B^T+1}86=ETR7@wJSUOZAST{IZoswTs82MD8*jy*{j_ArRPkvsUl0iuK>Z7aHd7l>-"
            b"$LBYHC~e9@)^_oaUysi}u_onjR`c=G?~4!KHGoRzkUm-cr;FnXfbut4z!eTQl+lTTIufc0+Hu|Wh4$})z~Qb_{?1k?"
            b"2GtyR`38g-1QDoJ)|Tw_4M%tP1+>WsN~&7M7*POh9TiVnF-"
            b"+nC*`d=4J(hKZ0icAjO4T8{OUV4R9~9ogV*XkwX*;Ag&ArJspm6gT%nSY9;}p9FA@%VU4jFycsJK6}zydKoyhr1Msr"
            b"3D{TBb4k-enTP7B^`ONKAapCdvvj5M?0xjim2ghd#%_9nP-OZ)|8F)(X%n!1fi6JTh=8ZC3HUmrxccyW__-"
            b"5H+V=D@*{=#aDr{J6zCj>p#{7(2f=}y34B!z$i*1XC==aSql5HK`Vr*_+#@^vK<NX?S!<SQ&5fP=qcS4r`*S3s7tc|"
            b"XP_C5tb$nRF94`x4F-c-Vdsc(^HXqFj=%#c6t!upV%%(~DZt|}j)p;qVh{+uJd7TJ&BG|mK{KKpdtl--"
            b"Oj>~Yig#N7N1*gIq4YI)J)9xIKl<<lgpYjdD?tye7O)W|&>$~;Eg-DrBTxXqHnFvgQ~ytfvH|t3WE|y3)PXt9GW4V)"
            b"sa6N>0Mg6O<bqCz{4O$#y-g9tgACtAIq<nBmduo&vM{C~NE45GIJx_!Tfx_Lgn}x+2^{3#v+gbH3;u-"
            b"YdrNK4YtYPCt-"
            b"><_4$urWf4<)&<e$2$JsVB?f&!c*O!3u`y(L4^y1#<=y~bI=upHn}=R_N<Z`iZ?KaoBRW+toPGbh9}hXZw9>m4ht%?"
            b"cIcPBjf#CmkD5McGcH7M-#pxtN%;88Jh4RYd-PM?63|B5iFtUIkU&b6#jE)7{V*wB0>gW4qpwFi6j_?2*&ksoz<>Q?"
            b"L33`W5I(Oc?21U#V@txuZ>Qn<-FhHfk>sKt&-"
            b"_fi^||fPzMulSw@&1V)#h^%qxP!I3gUhN2u>zme_SBFoau?n;|RiXMwEJV?QtFRT?R{Cg?{Ouoo|4#cOc0Edz%4jQ#"
            b"71PX$7?vWWGHGxIkoH|sp>RWAc*!#t?wSnfP!`IWrg3Jvv4AvBrZZ@A%a6glC&i~JnZli2(Qpn^F6L`y2c=%aC1lI|"
            b"SK!63t37Skhy_aYrD;P!(AO|deYl3h%Ow@P*@ygT`-"
            b"DXsm+zmVfyA9W*iUdHge%Sv6_>nYJIyPx5oun#FRFSMyjmyCPa-__0tIC+HZpscy`y&Ca)|FJ0T^bvm)my+SN4K(Ez"
            b"shc1F2Aw>3V2CjbXRi}nBSMP*b6gi{(Tzxz&E8G3+0d!I1Onzq_AQX>e7W&od9`@+R%iDILwu7Bim3(BL}%xdZ}t<w"
            b"dpihQ<%+F3e4sv9A<MR2Maw$xv=Q0DbV)8TcyDX_xYr;C`SV63Gz$t6*_b8b!Y!cqrAbBs(7m_(XT65u4|y3T&>Veu"
            b"25(uS19z7D>PckRT7;VVXIdw^x+jIkFYDziujEhl}M3+Jl)qjlAl@Rv=YQmtJS7JP-IclkiTcatOR4&Ia|voDuU}xr"
            b"UYs+39TZtBPqgmM~zW1zA1zPuhe1%Uae#5)3pS_7`vh+j$hG_V|w?F3{Vm|5aOW<oX?b7BlB~(sSB_B^t($ICt<5Z8"
            b"OR~Cry2JO0~@hy5T>>!b<D}N3~Oz|!M%jnbNdzgMt1((pJWg>>FI5@l(N*DRq}W3IZJR&+fD{NQ3xf#@_thWOel>v-"
            b"h=`(lam!N3LK#<>S;Ea--QTiCKE2$ykwnht~Xxi<ub?F15t{6oi9cg%*%d>W0;Fw=Nm~|(Rim`XAY3R)QX6~_s~Sb%"
            b"8SNa7W1>ZX>Ky4a=DBggW8b;!L;0rl(rUtkQU|hdiGf`>qsmhQ!wb8S5PffmvqTBgUv5pv%$DjQfHJ97I&0*)=0D62"
            b"`<)QlEG?K!fh#6KuRR2$)^Nfo3J+R9<Dgea$6jJ_9`FdBRR|!QN<Xeyy)|jgfjR8n|QvFsP+$f;L&Uz%sTEZjDv2g_"
            b"!<S;jf*|r6!&Qsk-gGG<%vOKYmK4|T()mfac^Z{HgO+T4(K?$s7zcg7Ara)nJCVAiesvhnf$1YI1QcnX-nh&2qq<G4"
            b"Kiw8A$1>t>d`VHaqrL7nQetNkhdx<F3>@@odJFT&!kU|imXr!Vg(VpOVXTX7gT6#bQZvsNl3_@oM-"
            b"8kD2_c^{e4E!xI&|7$_|1I7%33Iys_2%``>R2i2ndgNrKYS=7E!;4;?J{P0sEjSP11@;9Cn>KF$huvV^S%oRO>vK?1"
            b">GG{K`;U3yK#%2}oc$_QSX8j*v4U`~l6z=sW6hceW^5$P4&wpw2eJDR_(V@9qvVnuFX#z^B3TAjpqZ)kM+Ug%Sq8Kf"
            b"S@<^DZhA<#dWdiDtW64<O@<%m-|SWLpTbf8A}F?U(S<I_rb1pc~}UAC8+0iokKe_BZqm@+-dJCh2V12znt1b-"
            b"`F#ua)bhwD19yjD5_6=$eRQgZKQn^;<Kxji+UY*uH&Ra{6pGp-^;-UZ8U;CT|3D-"
            b"Kuf>TKNYa_r~t_r2rbfdCi0XUP;EO{OTbHdr!+$B{!hYp^a)`Psp`+YOiDa5*_d^Z7j34-"
            b"XH5U_KeogUP|%pG|_<Y=6ECruZ<N3}>TJfY4}qu#d<7e1HF7Ha=YX<#6;|-UV+ol05__GMjS$-"
            b"0OHhAOF7f`asCvWQ~UhGc=m}%h_@~^TR1R7#+-"
            b"}<N5S(HVa1h5YLvwa6Vavb2JYR_79gK8Xn>hL0N#OhkmH7@tVSMO~7E`32Fwbyg9bc>tC-v9-"
            b"l6}o5j^H*G((UGylDm+T-y|y8QhePT!Y-"
            b"feM>>ePz?o4^c2d!|8CmoDC1>C>Re%u!w{4A;Jgq;cz}3&JTm}d_Efu@obKlX#a4Cma~I7o=%XqGLNt;CSIneAe(qY"
            b"8TD3~R?OKNPcAS2xw!I9FRvC`u8=NF+eg$_HJhU;P+~mv4-e5eL`%G!16&8QU<eDqei%-12=p7G=@OPme)}_jI-5-"
            b"neLOoj&{k#6b1I2u-"
            b"t>so0kfXoyo!K)a8AeiGI6+^k0+DEAV8x@;LpeYbU6Vj+#iP1$s7c8G6{ldFbVeY;d~lSfFJR66pTkh{~$P=kF{k|y"
            b"=AckoyyN*2Lm)ehi|Z1{EsdXxb*b+{P@bdxjbK79bcRhGAfxKHc66kM}r~mkH!v7k>OtlfT7<Or#F{d_8uy|$HYnz("
            b"eTa}AI=t6)rqU37;SymD%rz&2F;*K)RL^`=ljU3_3h@Hl^tB~*=U)2C;b!eAJ@HqjeGBTG5Lm1xx+i@gCy68FIbQGF"
            b"eIrdv$iB-kv}zZ$W^jFVdebPxCv!CZ&!NcbyzfcG{eEc9zj~>5&x8+(B9um*7wm~Qrv@I(ul(@Kqjwz2Q0QGyuE})N"
            b"n>q=Lwb0z#tDk6^thMMR%Qum=+mP*1a&qk>`R9?z=e$%6M5O%P`hxTG2!neq^VxXR(MlRciselO<^vIN4c_ZIQ7~Nu"
            b"e768o8pJ#cX!d|M_O-!*m{JJpcK4fTaF<1mBT`3*oINENr}Mp(Eb)ll<U)6c&j$m?x-~-"
            b"rT#G0S?;vk5nLw!O2B^fPI!b}IwGYy@=N_l=T-"
            b"^!?w$0hRr=R>&wlhkmA)h5zW3<W73%ACs40t=qDr^ub=67Tp{<a}Pm#X^iS=MN&opKgHaDofe&d1lJ9Yn!`}BT_>JY"
            b"t>AGMt4AA-Vqs%bkGrBriA9>tuts%Sx}JpVGh9Y7}!#j)Mf&JJ*+Cp6r}hAZQ|vGIu##3DHNC=0%UcTmRJ+fY{~;yb"
            b"&H(<OQhk(*2|yX`~&=D#_2y495qoaZ=phyP=QN*nMBo5UOccm>RP;hi7<arrCR?`X&DwwmVUPNpNLp5^L}5B8<OK3`"
            b"p)oG*R>f9vArifyBs0#3Fe;N+^eOdKut<cj+wjtD`yjhITpwY%>a15&O%+IOtcKIg1jB+9iXuz{Hjd3_mzVqrBc1SJ"
            b"(}At?7FigWBV#-QBs&}j)l@x2pYf)s>uZ6!!qM9Buzyps%k8`ugWz-pC?g4FX5x@i<X5ZNwv<-"
            b"DGCmIfOiTT#JIMw*7;ij}3kpn@DH1yMi(&`(`iXjz!dHAKn8BFZ}euDxxtxRgL^;M0yF4N&l_RIObVaL1xfOXqZFqp"
            b"7G=FATt-2coV?`$}xWp{Lr#m3GBUFiGxel_f$0y>?qVzj}{st1KNyxo>!6yk19>WLfZ1Ol7_B^Ol8?>HRs!dIWC<{-"
            b"%zMloQ%?yUwnWtz3+epggS4l2sErRZ>nxr(g-"
            b"XUyjDN{In4acoF#wrKC~&)kpyj8?8(=xBH+7_j@`A0wOs;qNNNK?bDHxT@6TbG$2c6;Oa@7{roQ&g0$FU|5S;R-"
            b"^V827AOEZnPkDt3+=q4kx8b-mN#0(+!SQ-"
            b"#XzsV)g_2>f;X4^Rf13oJH32aW&`jYnB91}$KmH~EHTlL=VrRsA#Mx<O1HQ^EY@1yGB>2i)wQr9m24qoLUyr1v^HLq"
            b"wgp&USkg7SX)W^6Ns2Q|wvG}BE)tU;7}7rLgeJi&4%L^@(Jj3flND}RG`y{7!q5A$C+_G)z)@@s&%01V5}YXU<4qui"
            b"_~k?aa$Kbdlqs>KEuEk)VeGDlTcL^>;{m@J!EZH`a#8WcOIOf$WDV`Akjx<5En{kceK2Lfe{+?<zE0Y7LG6-"
            b"}Qv=eR!`ZGM2=Vnx(Ex_|gh2=AOveH7UnD1jtx7LYRf~d%s#NzgHBr#2RfVTVhYicAUmzDtGhsvkMl7Do7sl=HP`VV"
            b"sOrh4Gyfwi96f-fa=V;$v5QQNo4pjt@!X^!AEi^TLQXc>(>55!6D(}Ac2xB$a*ln4sA-T7SyDDp-QBKNkmysfeI3-"
            b"U?F19t6VM4VdZ}n<vmQ?19=1CXp5*2KWdlZM-"
            b"C92`zz~NiAXLf0cH6>xKTGOGisT`h%E_9MTOS&m&^>wo#N87EvJ!@RQF(L*ONGyj|Fkc(o6nYku$VLz}BN(KduaoTm"
            b"R$z)0O!pfs8V|uzJSjug5TQ)9w>MU^DdjzWnNF!oo2v_5DdKKjaX8A=FIT3G><S4CxhAZOrVM&lU7XC9YSN@aF0&Z1"
            b"4uF^n2fU{0_fGGrU2IYMmUYd&rMoXmPg368%WySD=`GhckD}b5QTnqQnI-"
            b"961#WiP2Q$0;_wGA3;b53<K{zj}(q8q%K^5(l=I7yG!6_2_9r-%2pQU+jH?62A&n4yC%7P)K35)rdxpQ0INKx-"
            b"9w%f?MPa>ZVNH++@Plf<j#ZOINC{b~Ftx__vyjCgM*;ZxQ<gT2#HLu15))u0#2U)n_%O5qPO^?oN+R5y=wCXBDQoBo"
            b"7C;xX=&}-x?j1t>KK~ShYg@vSCV%L-tERb5sqFepaow-u2x$39vKvbOR`ucSbjmmDS-"
            b"#y^=Gi=?0$a9f2Cs*Sz9f|D80o8PyFgM7fXKXa+4O!DFm)3IH6;a}bOD!xF0;OuLhWr^1hQ{(e%Va|unl`zl#^&!04"
            b"rr}muhTG#?$#>pdV>r-;Mmq)8sK`X1vL#;@4Z4{aA}MxO$EVli&}o6;YAoE+u5r*bp~h7sf}2FYC-"
            b"!{G}2DM8BDZ^Qo6g?;wHWVe6A;V0$lM5db*FeBM`lsTj*8KC4^w;3!F&d5`$dZhO(Zf2umLs=<jQGwZ2@FtU67tG%j"
            b"y!Cvjq98v{c$wpD1LA-5H`<erE56_z-rwkpb5hPcg9*HfAD|Ek6%YJYO2ZG_sa8mq3Q^yY>p)=!NC*MQZSr>7$U$GP"
            b"+%2l$Y5k3)u_zioR6Jo;m;37yEOObs<u*BQzcbz`XK@~qXgFoJAnSyI0g+Qd?j(0~L-8;-"
            b"|0>tRk?32rtAHq}>D^gDdrZkHFalgUlu-;@1AAYkKhz49=PUTGo6k&^FXx_;$Y@}j&NlG-D;G~iGR-$+~8X7@-"
            b"w&27LuN9uOEP>IL2UlcQk^vO*=Hj|WrFSwq)AQoliJwD^(VGWPzbj&JO+>6saakZAoN>i=6uTbmE`vkR(zD4LMrzr;"
            b"VlY{JL5<KsGw9;|~vCjSK;kkHYSGi|Mh((@yE8hVwh(DL*;5r*6pi!!SDE~_b*d5cWJ%$UJLW@^rF4nHeHfU{EEV88"
            b";w;}F!A}5><r$NiQ>Zf{8*EYBkD1ZLm(>ihklVO)APjOY3EK}tAZ*rnf$xF}VU<~-"
            b"FN>8c@_|8(%#wfi(<N9lusy4i@vn%R@6pf3qpGEw6P)%6_hDzT~fM|5t4ghrvUxUKeV@z8Dq$!`;>k``mC(Yhyu4D*"
            b"byiVVOC*7>bgOx@?dB;z#=kX-=ieIm1CZ-"
            b"3P)Vl+F@mj%a&Rw`wmjcDwj2b9C8O<$3|1WBa>IOI8rh%1ViOPLwobWW-2A80nE6k-"
            b"50$MTf^<mw#d>&<f1GF5k*D*_htI+T>^3!F609PqBxAYx6`O!~4yFi9VS41AlCH^<eQGw0^(W_;EMDE*BE5WN~0!u="
            b"{0`BB54v5n;7+eleD~o+?M6jR!2o%G-Nxn{rL2ATyRZ)SU;u&ULl9yQ-qN^pmF`}PM80Ayj6=c0H!`m8y<(u4kR?v5"
            b"Nk?k)roTjVwrr>Xi$!{-+qiXH-"
            b"G5Jvj>1BZWqYNZ$(}Kol?Mzqq_l$7c%}0l_mnl86;PMku_K53JcxiHuowyC}8e<C2z1~aV1`)j1>H6oN#TN9el0H|P"
            b"xQNzq1P=P^iEe)Vp(J*|UV&imbdq-"
            b";#7==`<q~lUsrz@1z!oKUM9fMq7KA3tjH?FB<3pQADG^Hul(rDFacr@cw?_tNA2`h5ju_<Abj)FlD$3?Y%p-"
            b"d>JxOa+xN#+*H2=(cNBQO<NVTH^6TuzrQBuP>Ozovw0R}iTjbJ#IvIdq618ULMUZ`+PFLnTLn_uZr!iNSQ#=sqpb&q"
            b"oLgFPHkrit}hUilDd4V4fZfiyk=V(4sK3hyp7fx5N^t<ZTP7-=GMtyk4rqs5D(H)EltQ`_pznpjaLwqd(Wp>=gmG6-"
            b"EYpKB;n&+%&@y2es@4{72Z*Ie8%7wMoE7_Ef(scC&04D-"
            b"0&@yb`=y)}A@QxFKM_s9b}?0+I)D7!WJ+EMiqXc@JM>E$Hyp9dl9=j+Q0@GMtrfR(mk<Ja75_J17N4Hsy7o&N_@hK;"
            b"i"
        ),
    ),
    "src/simsopt_jax/solve/fullspace.py": (
        "475cb63ddc183e343c1ae40faf7e0abf8bad5e6c288eabe38d31ce416e18cde4",
        (
            b"c-rkfX>;2~vfuS9u>6oB9gs^xw|24hWJPvdaV)Q;xN0}$LO^0D!3F^a0A*{h-~M_o%ng9DY$v-;RT2@?bM<uhbocc1"
            b"3<iU%yERI@ve?wfTNHU+#*5lp7NwV<+jN1(C92W{<+WF(`E`cIRUKa=Z@I~`akY*Y$g7I%7FCm-ovRz<<(m~M(?y(l"
            b"00%9u;>AsxBQLGII7y(RIfCDbSKputS}BWvpxj#*S-MD}4etkp*3b<u-XUM_U>J)mF5^1A#jU+V6&q`XGPJ1EBKKlw"
            b"uS5%!uItK!#?cz(c*NPA7jKfZj^`Pg><k8jogL_K<wenQQ*TNXMP9mE7iA4C<OK{D8mk!0qR3eD)g+!TIN({FWw<#C"
            b"l*DxmgR3eS6mO|Y(nalyA_|EBFC65uF3;g7EvWCvd|A<xG><didl9cta<y4QKRc}CZTvy{a{~3(FuTfoo3Gi3t8`Tr"
            b">pH^4*Ql7l-{~s-2P&({a#Nus%I3@KW}L$v5B_<A3!XoI^7QgzP3G!rpI@PBlhyR=%Mz{Q5+$Nk{r(bGX@zQ7eH~&="
            b"AtGrUYQegxs=|!SsTK3LcwI2XQG$#JK}MJ$DLra4%(6@T`DIzax~+Wm>m?>q`VHO2*#>@zmgx6o7S}b(#pejPqyjcc"
            b"^O{XV6uo;_N9z*sPDY3)uyc9#{rQil(d+Y9m(MO<c%e6t8EM?q9TT#y#<#P<&Z~=`uFfO1X7oBZp#;*Sjlk8$ZvQrR"
            b"2RqNtpPrumEqXnRXlrK|FRoslo?S(k&;EWMJ^tI(`K4%YfB*QnrN;NC!Ty1Orpq&!E#k%AJlQ{-CGp}g2x2rlIyyMs"
            b"od*Z;{9v&-I!J=W{vtluTLy>lZ@fQ0J_zE2U_U;bPiON1pqTdbI%pr>!O`)|25KMIe0RQkaCm%p9PI7x&JKfFvU?B&"
            b"vqf;cJl>0slf8qz#liA0IGP2^y~AY;#nb80{v1u`yYTf;8rN?7xQ=Fr$2L&=xE8y+heyGD?`ZER2=?}7NituegX!_^"
            b"@iIA_FBgZ0$9rghF^vxo<0G_tJU<GSyN8Rt!|4(wb98{mwX;Jcli-3^c*TcIE{#q|PiIdqqnA@mCh*tz<#f<|J$*jo"
            b"pW#pWdmz3C3Viffsxf{n$}a!<vS|$eElQqTy=qGE-ypno;=0taqfTf-r{KGaU??qRL0i4(u`l5fM9Zqc;uA}gO{F37"
            b"yJx3QUtC;XJv)moUOfLBo|E>8PWt7miytqpp1pXAA(|fyAr~*cdv*m3eEK|6;H(9MW@JyZVjgGdKgibhlI*A2=q}Iy"
            b"`qTM~t7lOE?-v(8;x@HUbOe`Ir+B!JPp?3rd>8$AdilfocLds2fJg9u%@#PM+kALcZjkR)Sy5Ny?-3j2S&`qO@*3qJ"
            b"rO6oS&?>qk$7NdIMR`%KPT*5%7S%<D$`~r2fUGX!nm`B{tyPr@tQGSjU#58q`#@Sn%Q($IQ8@ADMUibar%O<QK)lNH"
            b"xrO0k-65)InHUqa4*RsUub3dWMFyIc6J(pA!Jb}h$_0A7Nv=^%$F@w%s*dm$2-{b&Kvfli_JDxqqe?+*3`oC=X49Tv"
            b"U2w$;lpMfyE7beATt!67w3n~q59w;Nim1XGt<b7~&X-t`1tO6l@xrDs$D1ID0F8OI1oh|yKC2Dg6m=hvB)GAGEU`?o"
            b"6>haiSlr0eLw4m8O&&2Niisbr34#1M(FKw+@ER1|EK2hn2o9*cq*|{zLEL;vbt|b<uNSR3kboh&8(RelE1R4whn?$q"
            b"uMHH_#+p88$Ak#Xu>v0eSE0CE+`wSaS{m@RECEk!0+ADJk|tRD!}X&I^k!(J#wz&FWR$))nM~f8O-A#$hF;&NNqr-u"
            b"iPG5$l;?DvrKq$~A($y~R^Q1*>{E1I7VmL`CWb7P8cCoC9WL#-Xgx9IEF~-^4q(2jWT0(xD=R8D8QNahoGq)%U)M94"
            b"lEDyrLhU|cZWdWZ%uO))z}CwvG`?KiY)TKS(Q!6THn72iYCsuaLaexl<cBsMcw1mWE0LlADG{KuLJ_Rf$B+PF6=GiF"
            b"oeVFe!WQe|J<76Zu_<q3VzJtU8Xqm*8_;5dEf3l*tSqXPF<)=Ml;)J1VS~JMQ8~6`U|B%+CJ`#jqO>zE1_Q3#P<-R@"
            b"ClT=&z_Ph(lqe$xeFtcT3861Ae0V2d>l=W+DYC?db?l<3ZJb04#F9m;xV%mkY{OoZzAXURXt~5ri@er0zTR@Ap+LU;"
            b"CANTNY1`Pk5rfsa9+166ah1-~%&7^L@q1#HcI!+)4e2dKUy-vJv~JAUEbfNtx5@Y4#2;ptrb@+KzPKri9F{3z8U|EV"
            b"2lfCG@JU_vm4M4M1A+9%y4+Ot7#J4&w}flscPaxvtCZuN5)?*doTM9-&;WoaLauVIC^^cpIRVYng4Utdaej^G4A6^f"
            b"E0XDIN7wOc70a9~x2C`}5<!L#VuCQERrJM+1zOjb<0CQ(RYH)dR5xXcMW|kkKCFRz2m?53a<{RD2UgnDm&@kGCQo7q"
            b"hhu#%X4+m~@vqeBv2oGGS={w)|1-<tucEbwXccZ;rU7fxHA&;^0y~IS@w)QnsD6*&YvUOAz4gsqg%2xya)y9@h1G`@"
            b"@rbakgq#KYyvAntI)Y`oDL`k!p6JuEjPFd#9_gi$&7S5|Vj6HeJNWOqL?{B$mQAqauQ1ySTxgDKI%Ddj!a&h7MOjjX"
            b"Ka(%ihp7#F;EOVK$a0BmxD4iv&om{e%Qw^;w$+~aa{3*tPF_@Xx~Q6k`;MMD;S(pSjFE+r_{zf@5vshn_CQDRsBaQb"
            b"T$Blz-SXzh*OoAwu_#TcBvDle#?#y)BmL?yn4cAV5!Vs-R$1HY3m(IcNF7o4p%Wd!t#qPdN~Q%9OLD0VLvnc=b}a4m"
            b"0TozzU~)XoW?l3cZ?0Fcoug#@JiZ30cb&$$2O<M&HK`y;qA=0Z&xa$%rf;y?c;9_Uvg)M5>i0y`>MDH_qIP52P1;o}"
            b"HCMl5p18!Wz3*W;urIIKUDFT11l3KvMnLgns#bV66soagK!f4u`=`%6>PVCrf*arW6w9)z)lPHGdMZj+Yn?_zXsjo-"
            b"-1MHR<u)&-F%NQT%@w%h{tO{6Q9DOaly5EHov606jT_A}ypCeIOh>O=W<#wqW)-au&k-~tqTz)rRIbPwB#V}b@d|qI"
            b"cY@ul4Kpx%JiqqlafL8LXr{cthtOpzCwEwP5w={ydu0{vEyX$dAbG{MWz0{wZ&=Ny)*1}~Xy+Izl<O?S2GxhBDdaV_"
            b"dy(!m(NNu-G72C-#|C>7%{`ME?~J!lW!XbzUw>kVBcSlC-(D;v6BFkr7Ado(fRbZ~9HPle{Z>I`?B*ub>)WlPNT&Aw"
            b"GVv4a-`&)$nrAl3mLcp^Z9;U3Qa3d#XmnDtuA|tgSTj8-1uGICQ*W!jBME*+wcTPL2im+ofA#eI#o4(Rnnz<pdk8`}"
            b";0wdLuzPaC*<^~whrTTo!Dzx3kw6=mvEUV3Oak4eU@p<++39l{4x*g@4Xr^3;ZUFqWeD%t?^D>38ef6Od=rZz&B9gu"
            b"VHnS=;bJ=SXQMxP0}0=oav8%wISBiW0ns=B*c8a8DaAi%k=D%92IMK}gBFea_PB^}kfeBv^-#RH!P^57_+Wj3<u$+a"
            b"s$a``80_zj7>1p}6HPl+dYz+$0eajMh}*(LWA*D8D5S=X0@h!hUtaw5>g@ay@UEVl4m%?%!sF?5!q&DWq9U9fnP4`!"
            b"-2(yV^a#qG+9E3CR@yy=;dFB3yW=UsncwbA45xm(E0F_OA}Mg=nC3-{(&g3pOS}j}U>P5ZcqEeY-J4GK@qYsH-x>Yi"
            b"^nm?;mw%Y;^H0;mBlw6V1r5!6dGY+&*)y`*bbX}&4Z%`GsS8`e&}?Tjp>{Mg(Za(-&Gs0M5Qt-H0?XS3Uz+w%4y_?M"
            b"o+F$|1!hP`*ig{6K^@_2-#U6Dv?JV=pp;OI&=TA+(Sq4TRoh8zXwa3TEBUxf#i@SiuVO@t`~YmqrE2g8-U3W~*lTie"
            b"#`gkgnqp%M<vmY8H1QvykZDrOii;l0OYVO0t^s*}0;2)i7c}jmEWJ+Wjk}mh^6j{#iOlT9E=}CygEGprFQ<HElrPRG"
            b"(>6{C_P-FT6gj?tvIMf>#}w5Kz9#`4l_fytusxIG4ZgV%j;g#h*iRqFJ;5jEo`++ful)0sf2{m79Du>N0hRM^JSTd!"
            b"hj8BnEuO?%ux4Osd+c(7$IPR_=;QVu9O2)ahpbWfUw0#gn~t_(CflWgE|yXh|33V5^uNVVO#yYeVGeqKgYt%R*7PEy"
            b"zB|c>@4yMy&}rpMCOm0PL9z#iA~_A9zfy8Enn5$Q7fb<JFyg$XAs3W9S=XXX!MtFBkxN3+*(vbYcc=JD8#C-kROJv1"
            b"yGM`krOyw<*-rmTa~uwOjfS(?2wLDbji?oeuiz+Q$-|viX@G0`JNMa4&G<UOqR#lba29YFM}S?ZS>HCIZ3(>-hJ`cm"
            b"4%!L)K4EuZ;JZVM!yz7yM_pL~|AT)yV*fv2i`^1%thm$qMI5URw2qi#6}U6%SOs^)9fy3{?#N@Y5RSNG0boWQ)AslL"
            b"ofaL#!Z&_v=rLC@j9|BLZ-Orz278Bi;W}g9bPTmcy(x>^D+C!dvxd&yadP3obW5*TU$584JGNRQ<ig<SaI!ny3ucG="
            b"v;E`4qvHdAr&XX2a`fJN`tql7POr(!*C_DS3EcnU1~G++gkGX#oTF<Be)`;7WJQIwCQ0n_OPGq#6p_pwLKD~gO$*J>"
            b"Bp7Yc$P(XMVv)mOGJW)D9B5NkK#nHO$$%{w*(6n0C~|nP=eI^8D`LhTimZURVv!XfJs26rJM&$k$RY9C+9Hvm<Z$12"
            b"1R_h-^f+XUVc#BwEHKwM23Zw#eWH*Rf!8MtS?%JB$>y)3mK;Y~E@)Jn1)eIXpKb|@wcX|#TO|wHodhlE=ZK~}8Cvgk"
            b"jJ)w*%xj^8!Qd4F5tWlL6C8gi=TxE5Sik&28vNxK5pIG%GYHU?)gz7*Vf8E-(1fO3Nm=fd6GiziY5!|+r7&gl!4(_|"
            b"wP>P@@%iIx9AS5k8D-cc1MoK39;tp^M+$aEK8U@V%r^rB!xR9$v}`cy(?8yuy80*S2T1?Ry4<#Zqsir9lUJKH&P#$4"
            b"Id=|Q>3@>{|DpW%fYXC9Yd1~UuUJY{!6*V#A8pKsH6ubKzs6_F<AyFhwjri+sGf{XJn`*UbU9EOh;zXSeC`txV)<{5"
            b"KstEMZeyr17Niw5mb+IbYGAc3s3`!V{5ZsPkeZ1YJd%Ehk#sYTm?jMkb1FnQdrs6G?GLrf40;q<qp#saIVYCXIRvN9"
            b"OBZM4czv^|Wz05LFxZgjRScT?2t$z_DwE<51bS%#sgMt7lErs`kjW;8&fg8SX;Abe_1zj*(F!NzyXC5e**a;vSuizE"
            b"km^Cf0TNMCEUWNszMimsNCm<F;6f4~K?c%P0wF?5DiSbA&#1{LL-OB|($2_NezH)Dtvi?Nnf;@2(0>RdxI^w*?Fkp#"
            b"dYGv_XpKH?Y4K?n!lbt7$VX}j$tJI*)kYsc(*PZydKaZ&I^)|uB%b-_=^M`4O3gonUa-H{!pW*T4EU36^uV#dZqgFT"
            b"JOM3@MZC&m(vP@&ga<Xy=Hj}O12e)%f#L-qDOYGz9TEn^^1-B$p*&GvTN3;apxXN*X@JDl$g7}KZCl!gIcc%qMq9W8"
            b"Y&F1g3`O6V6!_{ba6HuYL=DUFz2VGvkpYvNy6h6z(C<oxng|iisuCnre=6Qv>ym}80_O5GWVtmqa@ZaZ?5lYp%v)nb"
            b"k~iRu2>&r&H;uB3EK;XlU4T4aA!0;j*gxEiO$L+hCsV8s)p2@bKC+39^${#KFH;d~prvWkz)Dvr^~WHm$Mi9l+&DCl"
            b"<3fe^t$iZiqrMn2FQRONpB6RXe&}Ob7ZZ2b9AeRSO!DC?R4LKQ(vnL`ItAk8aNVT>5!gW#*7^E;1|vE7A2#MfOC~^#"
            b"f_;u1MYsrI(wZ`GDEgX6pT%}77T6^!6iE>+mQ}Q#(mP5(JV_*CbUx>b(<B$7P2&#Yo0H$@@~L;Z@QCicekRXBb9x6o"
            b"7nq$Ausxn{x$5aPH-W4nhogyhY$<jpx))nfJCSzik@F_3z%$0fdz4<^;LvVz#0{cL3{xIE3Bv5`TIK}Sr}PE4cbgRT"
            b"&_uOwLBFg7h`{kCinAFZXh!(|X+;4viW!M9ed8%`P9!*M*s!Ss8^h5^Xw+H3f(aQFl8B&=eT7p6Zd5dc8XuWp!YS-T"
            b"jC%B~V<NPDluo=`u3vELp|Kncq#0fJZpwyE(V8HvQ2gne>Zf&X@>(#F&Lp}5_eJ;ab0$)C%brDo@&z7qk=2j^aWvBC"
            b"H?e`Uc}GBbJvJ}rw3PrIF*|97mL(t^sc05`<GU`OeaGzHH+|9X8*@omY#}*9R)~p4Sn&HhShvFNVVYJA9|_|9Ce93n"
            b"6!)c}q;-{i?<TIIc=Lf?mVB^9Q`YFqs5H(r6=z_*W4|3gZ5S_0<Vskf+y#|#R%jUlB3~2ApPe;SITV?gkKUm&M>yRs"
            b"M)TXW9-1(G-LM~xG!bVrp57biZP<NN)py(vcM7MM&=t27gcn7$O;&_UCJ@~PS%u^-k5?eOiZwp!N=!D;{3=6p(7xiT"
            b";5rs{LyR~gsJqgM3h~>Vsv#LHV&EszJWCxWuI2NXu4a~vKz7Yn4CLpN@eE{+^Z|a$#s;3p$6FiriGLzPL`@vQ&}Umw"
            b"p>(?!PW(CgYSC<C(TF4g+=dWx%&9Ln5Z@ZQzp`>J-CtDw?=wS8ctM&EXtHSQ_T{Y=m7PD6x#G?Y6KV)@*(+73{cbTx"
            b"s?b7N|0Q;ijxL7@)a*0AjeVt`S;Q0(z5zE=^+eJrw<XlEv-Z*Vn(04?d87_#Yy0RE@9{l$`gYritK)d3=(|QHK~gh{"
            b"Ei1RCdRtA-6}`9FbdLpQn&_HD2ydy0b|wvTa>wa<o!!Z-v4)wm=cB~VO|ZL&*K5grKu2oeI8tU{Ou&VmRyo??L<Pba"
            b"saWh7#Qm{I<4#GSiX7%;NAE+au5>a=Lr~vr9W*H3{3FXmY#Hh`zcmXa(}1`hCdMTzU-UkWy&?6^O?x|hmg>k@@a>TG"
            b"(r9_UK9iAZC)qJ;!KANuJklhqmYt@zalHLbMf01~tnsncOy;?&U{vE@y&rzKqBoFoq7=D~EuE1Gk1P{u-ld7Cpgw~Y"
            b"y&H<x&e)qY{c>U17){Il^uLhx$w=;`;g}?DdW9B%NE6dvv+%WL;iDMDzP=PD`)Y$tasB|{&vw6zujIUXBNl@{lGijz"
            b"f`2M=Nu0c`Tqo$oBF*WalT|o`HIpbW<dkh>W!vNga~L@%o`I~ba_cGRnKfeTB1azY0>_^-?ZKLfm8GF1R}^C@T;TVE"
            b"(5Mrm^orCHqkdSSY{|XUT3$k)yf&@Y_yP=(>YT>8GA60L(leb!zL4}TS>8i$6J>yEBpE4hMW*kDw`py7uOu@I=*$|!"
            b"pZ>f*bp>72bpz)AnBP6O)7t+ikp=f7vfwvOWJIZ-Z_+Gjbb7kF!;dsnl|AgF&oik`?SH#n=R}ExwnvLeA<;0pq6v^y"
            b"{G?o~Cce77-VH`f<;W_e!9Ky?XmMR`^is-<L<Nd-tqlJSUoEl6)w>j%g>3uRs)<-)X>>B}QtfXjB@HNXI!jsgWR{(9"
            b"C@6&%4eHH_ZdGn=fstWMbUTcWJG1G`(q(@t1{<21G_%zBg!a*cLG;1s!L!6MpC@Y)#Tc?7Oz6p_sMUbl&oFy9wd8*K"
            b"j?n7(Fq|LiTqq?)#XIbImu7>hEA*x#?_ZYBGo?O$Y8b2BitnDpQU9lDT@9TVZSr%RBodZD6JI_xYW;^*a~lI@6$!l)"
            b";e$2cjNok6l}%?hVE^K2#(bYlA!@|8Wo;T;xmjyWK;15%ZYq&+(&h{jGT25tBwwsOB$rC2_cy52?ubbA;@^)$O|H;J"
            b"q@e45!*JovToDIP$tDzx;y+S<>R?HKcv)rpU+L3|lF*HNZS|Uwh8l9Td223$&}0aT8XeI4R-Z3yNO%_-YI~&PZi6EQ"
            b"5Ze{(rO+qiPVvp!tc?wwGjmrF{dGzM>o9Ddkd9&4s)%9OO529vcC$E#;jSWv;gsHOV{yxF=UCiTyl=eDEun2Ngd;=R"
            b"3*E&2WtR;jN$bFRk6>BZouslJ%V}bS8aobLWua^qn>%`&x5<LaR++EXZe6|}eyiq?)Or0*(VI9mx5^3K_Z2EyiFOxH"
            b"mnMd^(5S#BCR#Y&x9la2_zSh)b{Znqo!9|<Mpi;!$EyA(R@m7n7A{abIb+MDi<AwRPI@(J^rADPmO&+|#neG2!tU!q"
            b"A1+{>9EfeG7}(7c6Ucdy-1S&uVolH*B%_yc1oBZdfZNL8UyumNP!fl}M2EhxA$>Yd)w~=Ux#9YqiFdkqr1{xBd%X62"
            b"m8mTpD3U(a=0>rqR4$5fs6g%DThU~T8se(bqB6LuwCwD4xk0v_*I<sOm?Sz^WMjp`Y%!gJJjrbuJ!U4I$tBt_uhqsh"
            b"E1qV)$*Cqyr;1hQw>UrK_cU5PASz<S%xxZpk?dPw>EQ3}VUYXkLHzcieVhYAJMo1Hw4MdG;Ja^+@7~o)L@G}9gYN*O"
            b"c<Ztv$f}uX5p<ThbRcGAi4wE;P-LVNrcu?TZ<JN(sRe)I4dq%bx_)T5VZN~-MNB{g1>D*SM@CA|#F)9WO;f8h2egfX"
            b"i~sp^$UQhxj?szj9TJDvh7AqRSJzh@job4}jhy}j{+e89d4Fj5h>1u;#gsS9K3n0I5<v?u-Hu&M9uB+MTH~;=mE&aj"
            b"8_GXA_1BhpI1IAZ@Y6mN+txs9>XVLm)raPmkvZ@7r<y_=Pq;KCCC9T&i@Gw^OM{Xdu!Hp0tjwmm4Z{|#%WTdD*58TK"
            b"PnvJxiRrhgjP!M6!aQvLvMPQ0j^;ZAzY#NSm58D8%doVVAS_Az1v%PW({soB9By`{ifU|v_B<r+b4^!Rj1mf?s4tGp"
            b"R(iG;^*&6i#iKOel5E92;nu)v({93Ph`Ll`d%B*kpgWqw$abJHF>^(0XCy@HZ$;EZOWh-}gC#ywPd(7T2N7_~&Fy9v"
            b">WGrI8rxk)wiYZYyNb*^2t9+eU7^U^m<($C0LfiN8F}3V5P(#!@O2cNcUD4;xo7t@`P*!KOIqG$rCkpa^m|i4+q6IC"
            b"Y*8XfJXPGd2(q}KU)0{193?5U`Gn+`6CAEFks+1UF-z2%&4y>x?ADUimfTM)XqYnqtgQPX+LHB0B5*R1t^(UbXgIMq"
            b"1THArXDdA#HBq}H5G!PDPqN}YDu)k6!=nl7VW@<2I|DO~2O10(Z5>c}@8Pz3+LX{QaeHy1d^f>4^;v50BI!<q6n_=+"
            b"Kh#SQs!lQcT*?HI*N#w87Q)sV>qwJr#udE)?uZv{GV>AlQk(W<mFCX8ab3E@Hdj$30#?yslC7A|WJ?N`YTloiEsb>3"
            b"wWU%QVOy%0lpPtQ&8o=X+bBKwa+DryLFvKoozjC&N)NtL`U6sW|4?d@5=B0d!XdzfzQ#oT=V6K@=LUN%NKJtv<qe`#"
            b"s;38%VrGT%#-hGuj7UD66C+j2j@9?$fa5Lpznel^?hoJp=FcilP{c9U@6q-xo@3{0*Y7oU`F(8*)|FSfbyVBx_Iu&H"
            b"3%A?9KCa-j^(pa|7sKy$-QeX^q^V(FJ&;Y=JYKw`NrQMcAZ*8Y@6-Cmi@7jg+~Cu?M9&oT3GNz@%thPYB<RnFk#obO"
            b"#g-^}!TIKe1mAn)TkO#joZLs`ZL-`|IRAiOYGKbLNZE4e2011#Z{So}l87SfqFz?8%o`gm@?C>OLxAcsXd=1keFr6;"
            b"`{OP=V59&(LK8U#92l&jvJo4eb8AjX)jPhGmF_<HwkN(aGf%mEFpju%%Vm(r?tQ^UEoZ1U`Nc00%gHh1)oQ}L^vQNK"
            b";+U1Cxwdmm6FJ?{i;9#q4k(@HC`y(>rOKaetkPa>WSn?@-1$xG5IS4vb!Khrz<HXx#bL5NQBDVmwN&k77i%+J)x~Q>"
            b"T56!}N8+500pH`?-*z<2_XLL+3H5&bu5}JA-~4v!H4j_uy|kP#+CXf*wlrY^^geuZ7wKD!Buzr)*H#B$HbZ<p{Ze+g"
            b"sn5v?PVl>?sASoHK&fLkbd&k9U7`DZpL4O-q{2RVsck1aUHWQ|vl=61t|QT&?v!phYkA-Umam5rpY%||u9^4ICws2n"
            b"7MZjZq4Bv-6No+ciH--h-UqqP9U9n+EPJo^3<?pf?71)gx?{AC@3JCJqTAV!h=n9w)IWn*^cicuu|4%QzpK%h32ORo"
            b"+xV6r39RC}Og|V&ZvH8Z$7*qdR&j(6+kqDbC*I}R_vb&JMz7CbT|T>bA^8;81Aiyp&vAtZ#ty_st#9rzzr(-#OnZwt"
            b">+|GIk+EPO1b&~A_<@1N=}iR#3EfN3Z>@qY#vLaKiBF}&9^VFH<itgS8J|9<NWK?Y)wXsN3oukzl<wL1EsIVUO<nrv"
            b"bs$&a@Ak<QpuJ-^+Ge$*+ue^q+^&i^_~@-RSe}EraXb4XAQrPlFYE=~*bDB1y<OS$$Dj7ou_rxtC$RVb(dJ;U"
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
