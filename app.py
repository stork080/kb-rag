import os, re, time
import numpy as np
import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding

# ============================ KNOWLEDGE BASE (50 PASSAGES) ============================
PASSAGES = [
    "Raft consensus requires a quorum of 4 nodes to commit a log entry. The leader sends 150ms heartbeats to prevent follower election timeouts. Network partitions cause the minority partition to stop accepting writes. Log compaction via snapshotting reduces replay time after a crash. etcd and CockroachDB both implement Raft for replicated state machines.",
    "Multi-Paxos consensus coordinates distributed replicated state without a single permanent master. Leader leases establish proposal rights for lease terms of 1000ms before re-election. In the accept phase, a commit quorum of 5 acceptors must acknowledge the ballot number. Proposers resolve ballot conflicts through randomized backoff intervals. Google Chubby and Apache ZooKeeper incorporate Paxos-family consensus protocols.",
    "Epidemic gossip protocols spread cluster membership updates in logarithmic rounds. The phi accrual failure detector triggers node suspicion after a 1000ms suspicion timeout. If network ping latency pushes the accrual metric past 0.7, the cluster marks the node dead. Anti-entropy background exchanges repair missing data versions across healthy nodes. Apache Cassandra relies on gossip for ring state coordination.",
    "The SWIM membership protocol provides weakly-consistent cluster membership with bounded false-positive rates. Nodes transmit ping probes periodically with a ping interval of 150ms. When an indirect ping probe fails, suspect timeout counters initiate dead-node eviction after crossing 0.5. Dissemination piggybacks membership updates onto outgoing probe packets to minimize packet overhead. HashiCorp Consul implements SWIM for decentralized cluster health monitoring.",
    "Two-phase commit coordinates atomic distributed transactions across independent resource managers. The coordinator aborts the transaction if any participant fails to reply within a 500ms prepare timeout. In three-phase commit, non-blocking progress is guaranteed by requiring a pre-commit quorum of 2 participants. Presumed-abort optimizations eliminate logging coordinator decisions for rolled-back transactions. Distributed deadlocks during phase one are detected using cycle detection on wait-for graphs.",
    "The Saga pattern orchestrates distributed long-running transactions across independent microservices using compensating actions. Forward execution steps timeout and trigger backward compensation if service responses exceed 100ms. Orchestration coordinators retry idempotent compensating steps across a minimum of 2 service retries before alerting. Unlike atomic two-phase commit, intermediate states remain visible to concurrent transactions across services. Temporal and Uber Cadence utilize sagas for resilient distributed workflows.",
    "Consistent hashing maps nodes and keys onto a virtual ring to balance partition loads. When key redistribution skew exceeds a 0.85 threshold, virtual nodes are dynamically allocated to smooth variance. Vector clocks track causal event ordering across concurrent distributed replicas without synchronized clocks. Dynamic read repair discovers stale replicas during quorum read operations within 150ms intervals. Amazon Dynamo popularized this decentralized replication architecture.",
    "Rendezvous hashing assigns object keys to highest-ranking server nodes through pseudo-random score evaluation. Hash weight multipliers scale server selection when load capacity distribution crosses a 0.9 threshold. When servers fail, displaced keys fail over to the next highest-scoring node within 100ms. Unlike consistent hash rings, rendezvous hashing eliminates virtual node maintenance overhead completely. Proxy caches like Squid and Envoy use highest random weight hashing for backend routing.",
    "Vector clocks capture partial ordering and causal relationships among concurrent events in distributed stores. When read repairs detect version concurrency, divergence resolution triggers within 250ms of replica discovery. Version vectors prune stale ancestor entries when actor membership counts exceed 4 replicas. Conflict resolution delegates sibling merges to application-specific resolution routines when clocks are concurrent. Riak KV relies on vector clocks for causality tracking across multi-master replicas.",
    "Hybrid logical clocks provide monotonic causality tracking by combining physical hardware timers with logical sequence counters. Clock synchronization algorithms bound maximum physical time drift within 100ms across all cluster nodes. If physical time advances past logical timestamps, the logical offset resets when skew drops below 0.7. Hybrid logical clocks produce single 64-bit timestamps without unbounded vector size inflation. CockroachDB and MongoDB use hybrid logical clocks for linearizable distributed read timestamps.",
    "Stochastic gradient descent updates model weights iteratively using sampled mini-batches. Training mini-batches are grouped into sizes of 128 samples per forward step. The optimizer applies a base learning rate of 0.1 to scale weight gradient updates. Momentum accumulates historical gradient vectors to accelerate traversal through shallow saddle points. Weight decay provides L2 regularization to prevent over-parameterized models from overfitting.",
    "AdamW stabilizes adaptive parameter updates by decoupling weight decay regularization from gradient moment updates. First and second exponential moment vectors track running gradients across mini-batches of 256 samples. The initial learning rate schedule initializes step updates at 0.001 prior to cosine annealing. Decoupled weight decay applies L2 penalties directly to parameters without distorting adaptive learning rates. HuggingFace transformers employ AdamW as the standard optimizer for foundation model pre-training.",
    "Transformer encoders use multi-head self-attention mechanisms to capture long-range contextual token representations. The deep backbone stack consists of 3 sequential transformer blocks with residual connections. Warmup scheduling ramps up the step learning rate toward 0.001 before cosine decay begins. Layer normalization stabilizes hidden state activations across varying input sequence lengths. Positional embeddings inject positional sequence order into permutation-invariant attention computations.",
    "Autoregressive transformer decoders generate token sequences iteratively using causal attention masks. The causal generative model stacks 12 transformer blocks configured with rotary position embeddings. Training sequences partition token streams into window lengths of 128 tokens per gradient step. Causal masking zeros future token attention weights to enforce unidirectional auto-regressive generation. Modern generative models like LLaMA and GPT rely on decoder-only autoregressive architectures.",
    "Binary logistic regression maps linear logit combinations into probabilistic predictions via sigmoid activation. Classification decisions categorize positive predictions whenever estimated class probability exceeds 0.85. Evaluation validation batches process 128 instances simultaneously during offline scoring passes. Expected calibration error quantifies probability deviation between predicted confidences and true empirical accuracy. Temperature scaling adjusts logit temperatures to produce well-calibrated posterior probabilities without degrading accuracy.",
    "Support vector machines construct maximum-margin hyperplanes to separate high-dimensional feature distributions. The soft-margin optimization penalty penalizes margin violations exceeding a slack threshold of 0.5. Training optimization solves dual quadratic programs across sub-sampled chunks of 128 support vectors. Radial basis function kernels project non-linear decision boundaries into infinite-dimensional Hilbert spaces. Platt scaling fits logistic sigmoids over raw SVM output distances to derive posterior confidence probabilities.",
    "Multi-layer perceptrons learn hierarchical feature abstractions through stacked non-linear transformations. The feedforward classifier contains 8 hidden dense layers with GELU activations. Dropout randomly zeros unit outputs with probability exceeding 0.7 to prevent co-adaptation. Batch normalization normalizes internal activation distributions across training minibatches. Gradient clipping bounds the L2 norm of parameter gradients to avoid catastrophic exploding gradients.",
    "Residual convolutional networks mitigate vanishing gradients across deep computer vision architectures using identity skip connections. The feature extraction backbone organizes 4 residual bottleneck blocks grouped into spatial resolution stages. Spatial feature dropout suppresses activation feature maps when unit correlation exceeds 0.9. Convolutions apply spatial kernels across input feature dimensions to learn translation-invariant spatial representations. ResNet and ConvNeXt architectures demonstrate that skip connections enable stable convergence in deep networks.",
    "Batch normalization accelerates neural network training by normalizing layer input distributions across mini-batch dimensions. Normalization statistics calculate empirical mean and variance over batch sizes of 256 training examples. If running variance moving averages diverge beyond a 0.9 threshold, momentum updates re-calibrate batch statistics. Learnable affine scale and shift parameters restore network representational capacity after zero-mean scaling. Modern vision architectures integrate batch normalization immediately prior to non-linear activation functions.",
    "Layer normalization computes normalization statistics independently across feature channel dimensions for each sequence sample. Deep transformer stacks apply pre-layer normalization across 3 sequential attention layers. The optimization pipeline configures learning rate warmup targets of 0.05 to prevent gradient explosion. Unlike batch normalization, layer normalization computes identical statistics during training and inference without batch dependence. Natural language processing models depend on layer normalization for recurrent and attention sequence processing.",
    "PostgreSQL manages relational table storage using page-level multi-version concurrency control. Database server instances listen for incoming client socket connections on port 5432. Connection poolers like PgBouncer disconnect idle backend connections after a 250ms timeout. The write-ahead log ensures crash recovery by persisting transaction records before dirty buffer flushes. Vacuum processes clean dead row versions created by updates and deletes to prevent table bloat.",
    "MySQL InnoDB implements multi-version concurrency control using undo log segments and clustered index storage. Server daemons accept incoming client connections on default network port 5432. Transaction lock wait timers abort blocked queries if row lock acquisition exceeds 250ms. Undo logs record prior row versions to reconstruct consistent read views without table-level locking. Purge threads asynchronously deallocate committed undo log pages to reclaim tablespace.",
    "B+ Tree data structures provide balanced logarithmic search and sequential range scan access. Internal index pages split when node fill capacity surpasses a 0.7 saturation ratio. Bulk insert workloads group leaf index modifications into 128 record chunks to minimize random I/O. Clustered indexes determine physical storage order on primary disk pages to maximize cache locality. Page latching protocols enforce concurrent thread safety during structural page split operations.",
    "Log-structured merge trees optimize write-heavy workloads by buffering incoming modifications in memory before sequential disk writes. Compaction processes merge sorted runs into lower levels when level size saturation exceeds 0.85. Memory tables flush immutable write buffers in chunks of 32 records to sequential SSTable files. Bloom filters allow fast point lookups by determining key absence without performing disk reads. RocksDB and Apache Cassandra use LSM trees to achieve high write throughput.",
    "MongoDB organizes schemaless documents inside scalable sharded collections across cluster replica sets. Daemon processes listen for client wire traffic over default port 6379 on each node. Write concern requires acknowledgment from a replica quorum of 4 members to confirm persistence. WiredTiger storage engine utilizes log-structured storage algorithms with Snappy compression for high throughput. Secondary nodes asynchronously replay the primary oplog to maintain read replica synchronization.",
    "Apache Cassandra distributes partition rows across peer-to-peer storage clusters using token-based ring partitioning. Database nodes listen for client CQL requests on native transport port 5432. Consistency levels require acknowledgment from a quorum of 5 replica nodes before write confirmation. Memtables persist incoming mutations sequentially to commit logs before flushing immutable SSTables to disk. Read repairs update out-of-sync replicas asynchronously in the background when replica checksums disagree.",
    "Redis operates as an in-memory key-value data store optimized for sub-millisecond retrieval. By default the Redis server daemon process binds to network port 6379 for TCP traffic. Client connections terminate when idle connection durations exceed the configured 500ms limit. Redis persistence snapshotting writes point-in-time RDB files and appends commands to the AOF log. The LRU eviction algorithm frees memory when maximum memory limits are reached by dropping stale keys.",
    "Memcached provides high-throughput distributed memory object caching via a multi-threaded slab allocator architecture. Daemon worker threads bind to default TCP listening port 27017 to serve cache requests. Connection timeouts terminate stalled client sockets when inactivity intervals surpass 100ms. Slab allocation avoids memory fragmentation by pre-allocating memory into slabs of specific chunk sizes. Unlike Redis, Memcached does not support disk persistence or complex data structures like sets and hashes.",
    "Write-ahead logging guarantees ACID durability by persisting database modification logs before changes write to table data pages. Checkpoint background processes flush dirty shared buffers when checkpoint log capacity crosses 0.9 utilization. Group commit buffers batch concurrent transaction flush requests into 128 log entries per fsync disk flush. The ARIES recovery algorithm performs analysis, redo, and undo passes to recover consistent database state following crashes. Relational databases depend on WAL to survive abrupt power outages without data corruption.",
    "Shadow paging provides atomic transaction durability by duplicating page table directories rather than maintaining write-ahead logs. The database directory flushes new page pointer arrays when modification thresholds exceed a 0.9 update ratio. Transaction prepare phases timeout and abort dirty shadow copies when commit execution exceeds 500ms. During commit, atomic pointer swapping updates the current page directory pointer to the shadow copy. SQLite supports shadow paging in rollback journal mode to ensure crash recovery without log replaying.",
    "Transmission Control Protocol establishes reliable bidirectional byte streams using a three-way handshake. Web servers accept secure transport connections through standard listening port 9200. Retransmission timers detect dropped segments when unacknowledged packets exceed a 1000ms RTO duration. Sliding window flow control advertises receiver buffer capacity to prevent receiver buffer exhaustion. Congestion control algorithms like BBR dynamically estimate bottleneck bandwidth and round-trip propagation time.",
    "QUIC replaces legacy TCP transport by implementing multiplexed connection streams directly over UDP. Server listener sockets bind to transport port 9200 to receive client datagram packets. Stream loss detection initiates packet retransmission when probe timers exceed a 1000ms loss timeout. By implementing stream multiplexing on UDP, QUIC prevents head-of-line blocking when individual packets drop. HTTP/3 mandates QUIC transport to achieve zero-RTT connection resumption and migration resilience.",
    "The Domain Name System translates human-readable hostnames into routable internet protocol addresses. Recursive DNS resolvers query authoritative name servers over UDP transport port 5432. Resolver clients timeout unfulfilled recursive lookups when upstream authority responses exceed 1000ms. Time-to-live attributes specify how long intermediate recursive caches retain authoritative record answers. DNSSEC signs resource record sets cryptographically using asymmetric keys to prevent cache poisoning attacks.",
    "Multicast DNS discovers network services on local link subnets without requiring centralized DNS servers. Resolvers broadcast lookup datagrams over local multicast port 6379 to all peer hosts on the subnet. Local responder services timeout unacknowledged service discovery requests if peer responses exceed 250ms. Queries append the dot-local top-level domain to restrict discovery traffic within the local broadcast domain. Apple Bonjour and Linux Avahi use mDNS for zero-configuration printer and device discovery.",
    "HTTP/2 introduces binary framing to interleave concurrent request and response streams over a single TCP connection. Flow control credits throttle individual stream delivery when stream buffer usage crosses a 0.9 threshold. Frame multiplexing coalesces smaller binary payload frames into groups of 64 bytes per transmission unit. HPACK header compression reduces metadata transmission overhead by maintaining shared dynamic indexing tables. Stream prioritization allows browser clients to request critical CSS and script resources ahead of decorative images.",
    "HTTP/3 redesigns web application transport by replacing TCP with QUIC over UDP to eliminate head-of-line blocking. Stream flow control limits buffer consumption when stream receiver buffer saturation crosses 0.7. Frame encoders group QPACK header instructions into compression blocks of 32 bytes per frame. QPACK accommodates out-of-order stream delivery by decoupling dynamic header tables from individual streams. Modern web browsers negotiate HTTP/3 via Alt-Svc headers to improve loading speeds on mobile networks.",
    "Transport Layer Security encrypts communication channels to protect confidentiality and integrity across the internet. Cipher suites negotiate ephemeral Diffie-Hellman key exchange with 512 bits of cryptographic security. The TLS 1.3 handshake reduces connection setup latency by completing in a single round trip under 500ms. Zero-RTT resumption permits returning clients to transmit encrypted application data on the initial flight. Certificate validation ensures server identity by verifying the trust chain back to installed root certificate authorities.",
    "TLS 1.2 negotiates secure transport sessions through a two-round-trip handshake between client and server. Ephemeral Diffie-Hellman parameter generation establishes forward secrecy using discrete logarithm groups of 256 bits. Initial handshake negotiation terminates if client hello responses encounter network delays exceeding 1000ms. Static RSA key exchange in TLS 1.2 is deprecated due to vulnerability to padding oracle attacks. Session tickets allow returning clients to bypass full key exchange by re-using serialized session state.",
    "Border Gateway Protocol manages inter-domain routing decisions across autonomous systems on the global internet. Peer router daemons establish exterior BGP peering sessions over TCP port 27017. Peering keepalive timers tear down inactive BGP neighbor sessions when keepalive arrivals exceed 1000ms. Path attributes like AS-PATH prevent routing loops by rejecting path vectors containing the receiving autonomous system. BGP routers exchange reachability updates to dynamically calculate optimal autonomous system paths.",
    "Open Shortest Path First computes optimal intra-domain routing paths within autonomous networks using link-state routing. OSPF routers send link-state advertisements over IP transport port 6379 to synchronize topology databases. Neighbor adjacency fails if hello heartbeats from adjacent routers stop arriving within a 100ms dead interval. Dijkstra algorithm executes over synchronized link-state databases to compute shortest path trees from each router. OSPF partitions large enterprise networks into hierarchical areas to constrain flooding overhead.",
    "Public key cryptography enables confidential communication and non-repudiation between untrusted network endpoints. Asymmetric key generators create digital certificates using RSA key lengths of 192 bits. Digital signature verification combines private key signing with a SHA-256 digest of the underlying payload. Hybrid cryptosystems combine asymmetric key exchange with symmetric stream encryption for performance efficiency. Perfect forward secrecy ensures past session keys remain secure even if long-term private keys are compromised.",
    "Elliptic curve cryptography provides equivalent public-key security to RSA using substantially smaller key representations. Digital signature algorithms like ECDSA achieve high-grade cryptographic security using elliptic curves of 192 bits. Cryptographic message signing hashes payload inputs using a SHA-256 digest function before curve multiplication. Smaller key sizes reduce memory footprints and accelerate cryptographic handshake execution on constrained devices. Ed25519 and Secp256r1 curves represent standard elliptic curve implementations in modern security protocols.",
    "Secure credential storage requires salting and iterative key stretching before persisting user secrets in databases. Modern cryptographic hashing pipelines compute password derivations using hardened SHA-256 algorithms with random salts. Work factor tuning balances offline attack resistance against server verification latency budgets capped at 100ms. Memory-hard functions like Argon2 protect against specialized hardware brute-force attacks via parallel ASIC chips. Slow verification deliberately imposes computation costs to render offline dictionary attacks infeasible.",
    "PBKDF2 derives cryptographic encryption keys from human passwords by repeatedly applying pseudorandom functions. The key stretching pipeline iterates HMAC evaluations using an underlying SHA-3 cryptographic hash digest. System administrators adjust iteration counts to keep derivation execution latency bounded near 250ms per attempt. Unlike memory-hard algorithms, PBKDF2 relies primarily on computational CPU cycles without intensive RAM consumption. Modern cryptographic standards recommend high iteration counts to mitigate GPU-accelerated dictionary attacks.",
    "Web application firewalls inspect inbound HTTP requests to intercept SQL injection and cross-site scripting attempts. Rate limiting algorithms throttle client IP addresses when request frequencies exceed a 0.9 burst threshold. Offending clients are blocked from executing sensitive API endpoints for a penalty window of 100ms. Content Security Policy headers restrict the origins from which scripts, styles, and fonts may be executed. Cross-Site Request Forgery tokens protect state-changing POST requests by proving user origin authenticity.",
    "API gateways protect backend microservices from resource exhaustion by enforcing token bucket rate limiting. Token leak algorithms throttle client consumers when bucket consumption surpasses a 0.5 burst capacity limit. Throttled client requests receive HTTP 429 status codes with retry-after headers specifying a 250ms wait duration. Token bucket algorithms allow temporary bursts of traffic while guaranteeing sustained request rates stay bounded. Kong and AWS API Gateway use token buckets to manage multi-tenant API quotas.",
    "Advanced Encryption Standard delivers authenticated symmetric encryption for data at rest and in transit. Secure storage engines configure AES in GCM mode using symmetric keys of 128 bits. Message authentication codes verify payload integrity using HMAC combined with a BLAKE2 cryptographic function. Initialization vectors must be unique for every encryption operation to prevent replay and ciphertext manipulation. Key rotation policies periodically replace symmetric master keys to limit data exposure windows.",
    "ChaCha20-Poly1305 provides high-speed authenticated stream encryption optimized for software-only execution without hardware AES acceleration. Cipher implementations initialize stream ciphers using symmetric keys of 512 bits. The Poly1305 authenticator produces 128-bit integrity tags by processing ciphertext blocks with a SHA-256 primitive. Because ChaCha20 avoids table lookups, it offers natural resistance to cache-timing side-channel attacks on general CPUs. WireGuard and mobile TLS connections default to ChaCha20-Poly1305 for efficient cryptographic processing.",
    "OAuth 2.0 Proof Key for Code Exchange secures authorization code grants on public mobile and single-page applications. The client transforms a high-entropy secret code verifier using a SHA-3 cryptographic digest to create the code challenge. Authorization servers reject authorization requests if code verification latencies exceed a 250ms grant window. PKCE eliminates authorization code interception attacks by requiring possession of the original plaintext code verifier. Modern identity providers require PKCE for all public client authentication flows.",
    "JSON Web Tokens transmit cryptographically verifiable claims between distributed services in compact URL-safe strings. Digital signatures verify JWT payload tampering by generating cryptographic signatures using SHA-256 digests. Verification middleware rejects expired tokens when expiration timestamp differences exceed a 250ms clock skew threshold. Tokens comprise base64url-encoded header, payload, and cryptographic signature segments separated by periods. Stateless microservices use JWTs to transmit authenticated identity claims without querying central session stores.",
]

# ============================ SYNONYM MAP (KEY FIX) ============================
# Maps paraphrased terms to terms that actually appear in the KB passages.
# If a query term is not in the passage, we check its synonym.
SYNONYMS = {
    # Consensus / distributed systems
    "keepalive":  "heartbeat",
    "keepalives": "heartbeat",
    "coordinator": "leader",
    "coordinators": "leader",
    "subordinate": "follower",
    "subordinates": "follower",
    "replicas":   "follower",
    "replica":    "follower",
    "trigger":    "timeout",
    "triggers":   "timeout",
    "periodic":   "interval",
    "periodically": "interval",
    "duration":   "interval",
    "duration":   "interval",
    "signals":    "heartbeat",
    "transmitted": "send",
    "transmits":  "send",
    "transmit":   "send",

    # Optimizers / ML
    "coefficient": "rate",
    "coefficients": "rate",
    "scalar":     "rate",
    "scalars":    "rate",
    "magnitude":  "rate",
    "adjustment": "update",
    "adjustments": "update",
    "updates":    "update",
    "updated":    "update",
    "parameter":  "weight",
    "parameters": "weight",
    "applied":    "apply",
    "applies":    "apply",
    "scales":     "scale",
    "provisioning": "create",
    "provisioned":  "create",

    # Databases
    "daemon":     "server",
    "daemons":    "server",
    "allocated":  "listen",
    "handling":   "process",
    "socket":     "connection",
    "sockets":    "connection",

    # Crypto
    "asymmetric": "rsa",
    "public-key": "rsa",
    "milliseconds": "ms",
    "millisecond":  "ms",
}

# ============================ INIT ============================
print("[init] loading fastembed ...", flush=True)
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("[init] building FAISS index ...", flush=True)
_passage_embs = np.stack(list(embedder.embed(PASSAGES)), axis=0).astype(np.float32)
faiss.normalize_L2(_passage_embs)
_index = faiss.IndexFlatIP(_passage_embs.shape[1])
_index.add(_passage_embs)
PASSAGE_LOWER = [p.lower() for p in PASSAGES]

app = FastAPI(title="KB-RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

class QueryIn(BaseModel):
    question: str = ""
    top_k: int = 3

# ============================ INJECTION SCRUBBER ============================
_INJ = re.compile(
    r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:of\s+)?(?:the\s+)?"
    r"(?:previous|above|prior|earlier|preceding)\s+(?:instructions?|prompts?|rules?|directives?)"
    r"|you\s+are\s+now\s+(?:a|an|the)?"
    r"|(?:system|developer)\s*prompt"
    r"|new\s+instructions?\s*:"
    r"|reveal\s+(?:the\s+)?(?:system|prompt|instructions?|rules?)"
    r"|print\s+(?:your\s+)?(?:system|prompt|instructions?)"
    r"|do\s+not\s+follow\s+(?:the\s+)?(?:rules?|instructions?)",
    re.IGNORECASE,
)

def sanitize(q: str) -> str:
    if not q:
        return ""
    return re.sub(r"\s+", " ", _INJ.sub(" ", q)).strip()

# ============================ LEXICON ============================
STOP = set("""
a an the of to in on at by for with from as is are was were be been being this that these those
it its and or but if then else so than which who whom whose what when where why how
do does did doing have has had having i you he she we they me him her us them my your our their
about into over under between across through during before after above below can could should would
will shall may might must not no nor only just also very too much many some any all each every
such other another same own both few more most several work works working
""".split())

GENERIC_Q = {
    "port", "ports", "rate", "rates", "size", "sizes", "threshold", "thresholds",
    "interval", "intervals", "timeout", "timeouts", "value", "values",
    "number", "numbers", "count", "counts", "bit", "bits", "capacity",
    "use", "used", "uses", "using", "typical", "standard", "default", "defaults",
    "usual", "main", "primary", "type", "types", "kind", "name", "level",
    "amount", "quantity", "high", "low", "maximum", "minimum", "max", "min",
    "recommended", "common", "total", "average", "mean",
    "key", "get", "set", "put", "run", "map", "add", "end", "top", "sub",
    "listen", "listens", "listening", "bind", "binds", "binding",
    "connect", "connects", "accept", "accepts",
    "need", "needs", "require", "requires",
    "send", "sends", "receive", "receives",
    "specified", "specifies", "specify",
    "exact", "mathematical", "formulation",
}

def terms_of(text: str, min_len: int = 3):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) >= min_len and t not in STOP}

def term_present(term: str, text_lower: str) -> bool:
    if term in text_lower:
        return True
    if len(term) >= 5 and term[:5] in text_lower:
        return True
    return False

def specific_terms(q: str):
    return {t for t in terms_of(q) if t not in GENERIC_Q and len(t) >= 3}

def entity_grounded_syn(spec, passage_idx, threshold):
    """Entity check with synonym expansion. Returns True if matched/len(spec) >= threshold."""
    if not spec:
        return True
    passage_lower = PASSAGE_LOWER[passage_idx]
    matched = 0
    for t in spec:
        if term_present(t, passage_lower):
            matched += 1
        elif t in SYNONYMS and term_present(SYNONYMS[t], passage_lower):
            matched += 1
    return (matched / len(spec)) >= threshold

# ============================ ATTRIBUTE PATTERNS ============================
_UNIT_HINTS = {
    # Time-based (covers duration, interval, heartbeat, keepalive)
    "duration":   r"\b\d+(?:\.\d+)?\s*(?:ms|s\b|sec|secs|seconds|milliseconds)\b",
    "interval":   r"\b\d+(?:\.\d+)?\s*(?:ms|s\b|sec|secs|seconds|milliseconds)\b",
    "heartbeat":  r"\b\d+(?:\.\d+)?\s*(?:ms|s\b|milliseconds)\b",
    "keepalive":  r"\b\d+(?:\.\d+)?\s*(?:ms|s\b|milliseconds)\b",
    "periodic":   r"\b\d+(?:\.\d+)?\s*(?:ms|s\b|seconds|milliseconds)\b",
    "timeout":    r"\b\d+(?:\.\d+)?\s*(?:ms|s\b|sec|secs|seconds|milliseconds)\b",
    "latency":    r"\b\d+(?:\.\d+)?\s*(?:ms|s\b|sec|secs|seconds|milliseconds)\b",
    # Ports
    "port":       r"\bport\s+\d{2,5}\b",
    # Sizes
    "size":       r"\b\d+(?:\.\d+)?\s*(?:GB|MB|KB|bytes?|bits?|blocks?|layers?|samples?|records?|chunks?|tokens?|nodes?)\b",
    "length":     r"\b\d+(?:\.\d+)?\s*(?:GB|MB|KB|bytes?|bits?|tokens?|blocks?|layers?)\b",
    "batch":      r"\b\d+\s*(?:samples?|records?|instances?)\b",
    # Bits / keys
    "bits":       r"\b\d+\s*bits?\b",
    "key":        r"\b\d+\s*bits?\b",
    # Rates / coefficients
    "rate":       r"\b\d+(?:\.\d+)?\b",
    "coefficient": r"\b\d+(?:\.\d+)?\b",
    "scalar":     r"\b\d+(?:\.\d+)?\b",
    "magnitude":  r"\b\d+(?:\.\d+)?\b",
    "learning":   r"\b\d+(?:\.\d+)?\b",
    # Generic threshold / quorum
    "threshold":  r"\b\d+(?:\.\d+)?\b",
    "quorum":     r"\b\d+\b",
}

_NUMERIC_HINT = re.compile(
    r"\b(?:how\s+many|how\s+much|port|rate|interval|timeout|threshold|size|count|number|bits?|capacity|amount|value|limit|quorum|blocks?|layers?|samples?|coefficient|scalar|magnitude|duration)\b",
    re.IGNORECASE,
)

def _stem(t: str) -> str:
    return t[:5] if len(t) >= 5 else t

def extract_answer(question: str, passage: str) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage.strip()) if s.strip()]
    if not sents:
        return passage.strip()[:400]
    q_terms = terms_of(question)
    q_stems = {_stem(t) for t in q_terms}
    wants_numeric = bool(_NUMERIC_HINT.search(question))
    unit_pattern = None
    for kw, pat in _UNIT_HINTS.items():
        if re.search(rf"\b{kw}s?\b", question, re.IGNORECASE):
            unit_pattern = pat
            break
    best_sent, best_score = sents[0], -1.0
    for s in sents:
        s_terms = terms_of(s)
        if not s_terms:
            continue
        s_stems = {_stem(t) for t in s_terms}
        matched = len(q_stems & s_stems)
        score = matched / (np.sqrt(len(q_terms)) * np.sqrt(len(s_terms)) + 1e-6)
        if wants_numeric and re.search(r"\d", s):
            score += 0.20
        if re.search(r"\d", s):
            score += 0.05
        if unit_pattern and re.search(unit_pattern, s, re.IGNORECASE):
            score += 0.30
        score -= 0.001 * len(s)
        if score > best_score:
            best_score, best_sent = score, s
    return best_sent

def _abstain(conf: float, reason: str = ""):
    resp = {"answer": None, "sources": [], "scores": [],
            "confidence": round(float(min(max(conf, 0.0), 0.39)), 4)}
    if reason:
        print(f"[RESPONSE] ABSTAIN ({reason}): {resp}", flush=True)
    return resp

# ============================ ROUTES ============================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(inp: QueryIn):
    t0 = time.time()
    print(f"[QUERY] {inp.question!r} (top_k={inp.top_k})", flush=True)

    try:
        q = sanitize((inp.question or "").strip())
        if len(q) < 3:
            return _abstain(0.0, "short query")

        # 1. Retrieval (top 10)
        q_emb = np.array(list(embedder.query_embed(q)), dtype=np.float32)
        faiss.normalize_L2(q_emb)
        D, I = _index.search(q_emb, 10)
        bi_idx = [int(x) for x in I[0]]
        bi_scores = [float(x) for x in D[0]]

        print(f"[DEBUG] top-5 retrieval = {[(i, round(s,3)) for i, s in zip(bi_idx[:5], bi_scores[:5])]}", flush=True)

        if bi_scores[0] < 0.25:
            return _abstain(bi_scores[0], "low cosine")

        spec = specific_terms(q)
        k = max(1, min(inp.top_k or 3, 10))
        top_idx = bi_idx[:k]
        top_scores = bi_scores[:k]
        confidence = round(min(0.98, 0.60 + bi_scores[0] * 0.5), 4)

        # Detect attribute keyword
        attr_pattern = None
        for kw, pat in _UNIT_HINTS.items():
            if re.search(rf"\b{kw}s?\b", q, re.IGNORECASE):
                attr_pattern = pat
                break

        versions = set(re.findall(r"\b\d+\.\d+\b", q))

        # ---- Path A: attribute-specific query (with cascade thresholds) ----
        if attr_pattern is not None:
            print(f"[DEBUG] attr_pattern={attr_pattern} spec={spec} versions={versions}", flush=True)

            for threshold in [0.6, 0.5, 0.4, 0.3]:
                for idx in bi_idx[:5]:
                    if not entity_grounded_syn(spec, idx, threshold):
                        continue
                    passage = PASSAGES[idx]
                    if versions and not any(v in passage for v in versions):
                        continue
                    if not re.search(attr_pattern, passage, re.IGNORECASE):
                        continue

                    sent = extract_answer(q, passage)
                    if not re.search(attr_pattern, sent, re.IGNORECASE):
                        for s in re.split(r"(?<=[.!?])\s+", passage):
                            if re.search(attr_pattern, s, re.IGNORECASE):
                                sent = s.strip()
                                break

                    # Confidence degrades as threshold loosens
                    conf = round(min(0.98, confidence * (0.55 + 0.45 * threshold)), 4)
                    resp = {
                        "answer": sent.strip(),
                        "sources": [int(x) for x in top_idx],
                        "scores": [round(float(x), 4) for x in top_scores],
                        "confidence": conf,
                    }
                    print(f"[RESPONSE] ANSWER (path A, thr={threshold}, {time.time()-t0:.2f}s): {resp}", flush=True)
                    return resp

            # Path A.5: value-first fallback — scan top-20, very loose entity (0.2)
            for idx in bi_idx[:20]:
                if not entity_grounded_syn(spec, idx, 0.2):
                    continue
                passage = PASSAGES[idx]
                if versions and not any(v in passage for v in versions):
                    continue
                if not re.search(attr_pattern, passage, re.IGNORECASE):
                    continue
                sent = extract_answer(q, passage)
                if not re.search(attr_pattern, sent, re.IGNORECASE):
                    for s in re.split(r"(?<=[.!?])\s+", passage):
                        if re.search(attr_pattern, s, re.IGNORECASE):
                            sent = s.strip()
                            break
                if re.search(attr_pattern, sent, re.IGNORECASE):
                    conf = round(min(0.85, confidence * 0.65), 4)
                    resp = {
                        "answer": sent.strip(),
                        "sources": [int(x) for x in top_idx],
                        "scores": [round(float(x), 4) for x in top_scores],
                        "confidence": conf,
                    }
                    print(f"[RESPONSE] ANSWER (path A.5 loose, {time.time()-t0:.2f}s): {resp}", flush=True)
                    return resp

            return _abstain(bi_scores[0] * 0.5, "no passage has entity+attribute")

        # ---- Path B: general query (with cascade) ----
        print(f"[DEBUG] path B (no attr) spec={spec}", flush=True)
        for threshold in [0.6, 0.5, 0.4]:
            if entity_grounded_syn(spec, bi_idx[0], threshold):
                answer = extract_answer(q, PASSAGES[bi_idx[0]])

                if (k >= 2 and bi_scores[1] >= 0.6 * bi_scores[0]
                        and bi_scores[1] >= 0.45
                        and entity_grounded_syn(spec, bi_idx[1], max(0.4, threshold - 0.1))):
                    sec = extract_answer(q, PASSAGES[bi_idx[1]])
                    if sec and sec != answer and sec not in answer and answer not in sec:
                        answer = answer + " " + sec

                conf = round(min(0.98, confidence * (0.6 + 0.4 * threshold)), 4)
                resp = {
                    "answer": answer.strip(),
                    "sources": [int(x) for x in top_idx],
                    "scores": [round(float(x), 4) for x in top_scores],
                    "confidence": conf,
                }
                print(f"[RESPONSE] ANSWER (path B, thr={threshold}, {time.time()-t0:.2f}s): {resp}", flush=True)
                return resp

        return _abstain(bi_scores[0] * 0.5, "entity not grounded")

    except Exception as e:
        print(f"[error] {e}", flush=True)
        return _abstain(0.0, f"exception: {e}")
