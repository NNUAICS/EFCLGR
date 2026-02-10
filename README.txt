#EFCLGR

Please run the code "main_EFCLGR.m" for the implemetation of the EFCLGR method. 

1.“compute_L” is the function used to calculate the covariance matrix C (In the original text, it is S_t.) and the Laplacian matrix L.
2.“EFCLGR” is the main solving function.
3.“L2_distance_subfun” is a subfunction required by EFCLGR, used to compute H.
4.“SimplexQP_ALM” is a subfunction required by EFCLGR, used to solve the membership matrix U.
5.“EProjSimplex” is a subfunction required by SimplexQP_ALM.