# References

보고서 References 섹션과 README "References and external code"에 쓰는 목록입니다. 담당: 강민수.

## 목록

[1] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank citation ranking: Bringing order to the Web," Stanford InfoLab, Stanford, CA, USA, Tech. Rep. 1999-66, 1999. [Online]. Available: http://ilpubs.stanford.edu:8090/422/

[2] Z. Gyöngyi, H. Garcia-Molina, and J. Pedersen, "Combating web spam with TrustRank," in *Proc. 30th Int. Conf. Very Large Data Bases (VLDB)*, Toronto, ON, Canada, 2004, pp. 576–587.

[3] C. Castillo, D. Donato, L. Becchetti, P. Boldi, S. Leonardi, M. Santini, and S. Vigna, "A reference collection for web spam," *ACM SIGIR Forum*, vol. 40, no. 2, pp. 11–24, Dec. 2006, doi: 10.1145/1189702.1189703.

[4] "Web Spam Collections" (WEBSPAM-UK2007). [Online]. Available: http://chato.cl/webspam/datasets/ . Crawled by the Laboratory of Web Algorithmics, Univ. of Milan, http://law.di.unimi.it/ . Data retrieved Nov. 2026.

[5] P. Boldi and S. Vigna, "The WebGraph framework I: Compression techniques," in *Proc. 13th Int. Conf. World Wide Web (WWW)*, New York, NY, USA, 2004, pp. 595–602, doi: 10.1145/988672.988752.

[6] T. H. Haveliwala, "Topic-sensitive PageRank," in *Proc. 11th Int. Conf. World Wide Web (WWW)*, Honolulu, HI, USA, 2002, pp. 517–526, doi: 10.1145/511446.511513.

[7] Z. Gyöngyi, P. Berkhin, H. Garcia-Molina, and J. Pedersen, "Link spam detection based on mass estimation," in *Proc. 32nd Int. Conf. Very Large Data Bases (VLDB)*, Seoul, Korea, 2006, pp. 439–450.

[8] V. Krishnan and R. Raj, "Web spam detection with Anti-Trust Rank," in *Proc. 2nd Int. Workshop Adversarial Inf. Retrieval Web (AIRWeb)*, Seattle, WA, USA, 2006, pp. 37–40.

[9] A. Z. Broder, "On the resemblance and containment of documents," in *Proc. Compression and Complexity of Sequences*, Salerno, Italy, 1997, pp. 21–29.

[10] D. Gibson, R. Kumar, and A. Tomkins, "Discovering large dense subgraphs in massive graphs," in *Proc. 31st Int. Conf. Very Large Data Bases (VLDB)*, Trondheim, Norway, 2005, pp. 721–732.

[11] L. Becchetti, C. Castillo, D. Donato, S. Leonardi, and R. Baeza-Yates, "Link-based characterization and detection of web spam," in *Proc. 2nd Int. Workshop Adversarial Inf. Retrieval Web (AIRWeb)*, Seattle, WA, USA, 2006, pp. 1–8.

[12] J. Leskovec, A. Rajaraman, and J. D. Ullman, *Mining of Massive Datasets*, 3rd ed. Cambridge, U.K.: Cambridge Univ. Press, 2020, doi: 10.1017/9781108684163.

[13] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2. (사용 버전 2.5.5)

[14] P. Virtanen *et al.*, "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261–272, 2020, doi: 10.1038/s41592-019-0686-2. (사용 버전 1.18.1)

[15] W. McKinney, "Data structures for statistical computing in Python," in *Proc. 9th Python in Science Conf. (SciPy)*, Austin, TX, USA, 2010, pp. 56–61, doi: 10.25080/Majora-92bf1922-00a. (pandas, 사용 버전 3.1.1)

[16] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011. (사용 버전 1.9.1)

[17] A. A. Hagberg, D. A. Schult, and P. J. Swart, "Exploring network structure, dynamics, and function using NetworkX," in *Proc. 7th Python in Science Conf. (SciPy)*, Pasadena, CA, USA, 2008, pp. 11–15. (사용 버전 3.7)

[18] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55. (사용 버전 3.11.2)

## 어디에서 인용하나

| 번호 | 내용 | 보고서 위치 |
|---|---|---|
| [1] | PageRank | 1. Introduction, 3. Methods |
| [2] | TrustRank, seed 선택 | 1. Introduction, 3. Methods |
| [3], [4] | WEBSPAM-UK2007 데이터셋과 라벨 (데이터 설명 논문 + 데이터 배포처) | 2. Data |
| [5] | host 그래프 수집·압축 (WebGraph) | 2. Data |
| [6] | Topic-specific / Personalized PageRank | 3. Methods (PPR) |
| [7] | Spam Mass | 3. Methods (TrustRank 점수 ②) |
| [8] | Anti-TrustRank (스팸 seed 역방향 전파) | 3. Methods (PPR 역방향) |
| [9] | MinHash (Jaccard 근사) | 3. Methods (LSH) |
| [10] | 링크 집합 shingling으로 밀집 구조(링크 농장) 찾기 | 1. Introduction 또는 3. Methods (LSH 동기) |
| [11] | 같은 데이터 계열에서 링크 기반 스팸 탐지 선행연구 | 1. Introduction (선행연구) |
| [12] | 교재 (알고리즘 정의 근거) | 3. Methods |
| [13]–[18] | 사용 소프트웨어 (버전은 requirements.txt) | 3. Methods 또는 4. Experimental design (실행 환경) |
