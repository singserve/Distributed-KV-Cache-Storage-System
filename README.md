# Distributed KV Cache Storage System: Extending GPU KV Capacity with CPU/DPU DRAM Pooling for cost efficient Inference
## abstract
KVCaches are tremendously stored on GPU for fast LLM prefilling prefix reuse and decoding acceleration. Many storage systems such as LMCache, Mooncake and HiCache are integrated in inference system to offload KVCaches for future reuse. However, most of them offload data to a low level, such as CPU pool or remote backend, and upload when data are used. This long-way vertical and cross-CPU travel of KVCaches in the storage system will take a lot of time, which
makes some SLOs hard to meet. To address this, VCache, a VRAM-level KVCache system is presented in this article. VCache makes following contributions: 
(i) distributed VRAM pool with direct data transfer to enable fast in-GPU data feed
(ii) multi-level storage system enables offloading to mooncake storage backend as fallback. 
(iii) available system APIs such as store, retrieve and lookup.

## motivaton
The experiment below shows that the delay of cross-GPU transfer via DRAM is far longer (usually dozens of times) than that directly via NVLink bypassing DRAM.Thus,whenutilizingtheadvantageofprefix-caching, transfer directly across GPU VRAM is preferred.

<div align=center>  <img src="./assets/vramvsdram.png" width=50%></div>

The solution in this article is to build up a VRAM pool like CPU/DRAM/SSD pool like mooncake store. The VRAM pool allocates pieces of memory segment in each GPU distributively, and manages the metadata of the KVCache pieces in each GPU centrally using a separate VRAM metadata manager, just like how mooncake store does using a master node. The KVCaches are stored, managed, andretrieved within VRAMpool,which makes it fast to feed the inference applications.

## overview of vcache
<div align=center>  <img src="./assets/vcache_architecture.jpg" width=70%></div>

## interfaces

| External APIs      | Description |
| ----------- | ----------- |
| Lookup(tokens)-> int| Look up in the VCache system for stored KVCache, return the number of hit tokens. This function first searches the VRAM pool; if hits, it returns the hit token count. Otherwise, it falls back to mooncake backend. |
| Store(tokens, mask, kvcaches, slot_mapping, offset)-> None   | Store KVCaches with token ids into the VCache system. This function stores KVCaches in both VRAM and the mooncake store backend with slot mapping.        |
|Retrieve(tokens, mask, kvcaches, slot_mapping)-> torch.Tensor | Retrieve hit KVCaches for a sequence of tokens and upload the data to the kvcaches parameter using slot mapping. This function first retrieves hit data from the VRAMpool. If no hits, it falls back to retrieve from the mooncake store backend. Returns a boolean mask indicating retrieved tokens.|
| contains(cache_key) -> int | check if the cache key exists in the cache engine. 0 if exists in GPU VRAM, 1 if exists in storage backend, -1 if not found  |
| get_stats() -> Dict | Get cache engine statistics and status   |

## NOTE: THE PROJECT SRC IS STILL UNDER REVISION

## file organization
**VCache**: src   
**test**: scripts to test system functions  
**integration**: vllm kv connector and factory registration  
**log.txt**: vcache engine log output when run cross gpu store and retrieve  
**server_log.txt**: metadata server log   

## quick start
### setup
| module | version |
| -------| --------|
| lmcache | 0.3.9.post3.dev3 |
| torch | 2.9.1+cu128 |
| vllm | 0.11.1rc7.dev147+gda14ae0fa.cu124 |
| Python | 3.10.12 |
| Mooncake store | 2.0.0 |
| Transfer engine | 0.3.7.post2 |

### run scripts to test VCache engine store() and retrieve()
1. lmcache, vllm and mooncake are buit from source with editable option.  
2. put `Vcache` directory in `LMcache/lmcache`  
3. edit `system_config.yaml` to configure system settings  
4. start VRAM pool metadata manager  
   `cd VCache/vram_metadata_server && VCACHE_LOG_LEVEL=DEBUG python3 start_ipc_server.py`  
5. run mooncake master  
   `mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=8080`  
6. set config file path and run test scripts  
   `cd tests && VCACHE_LOG_LEVEL=DEBUG python3 test_cross_gpu_store_retrieve.py`  
7. check log and metadata server log  
### integration with vllm (UNDER REVISION AND TEST)
1. set up VCache system and vllm, lmcache, mooncake
2. replace `factory.py` in vllm project with `factory.py` in `integration` to register kv connector
3. put kv connector in `integration` in 'kv_connector' repository in vllm project
