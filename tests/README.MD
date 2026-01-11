log shows the output when running `test_cross_gpu_store_and_retrieve.py`  
1. generate mock paged kvcache  
2. initialize vcache engine0 using gpu0  
3. store kvcache in vram on gpu0  
4. initialize vcache engine1 using gpu1  
5. retrieve kvcache in vram on gpu1  
6. compare stored and retrieved kvcache  

