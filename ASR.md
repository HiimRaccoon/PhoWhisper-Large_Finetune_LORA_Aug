# Thiết kế ASR độc lập: PhoWhisper-large + LoRA + augmentation

**Trạng thái:** đặc tả để xây dựng một project ASR mới; lần train full-data chưa
được triển khai hoặc chạy. Có thể sao chép riêng file này sang một thư mục chứa
dataset và dùng làm yêu cầu cho người hoặc agent xây dựng. Tên file code và cấu
hình trong tài liệu là **đề xuất cần tạo**, không phải lệnh đã có sẵn.

Mục tiêu là fine-tune ASR tiếng Việt trên toàn bộ **1.545 source sample**, dùng
audio current và audio của từng user-turn trong history. Không chia Validation
trong run mới. Mỗi lượt train chọn một audio view: 50% vào nhánh clean, 50% vào
nhánh augmentation theo chính sách E4. Sau khi xuất model ASR offline, chạy model
đó trên audio gốc để tạo transcript cho từng sample Train phục vụ fine-tune LLM.

## 1. Sơ đồ luồng dữ liệu và huấn luyện

```text
+--------------------------------------------------------------+
| 1.545 SOURCE SAMPLES                                         |
| Fit + Validation cũ, đưa toàn bộ vào train                   |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| USER AUDIO: CURRENT + HISTORY                                |
| 5.036 lượt theo thống kê nguồn trước audit/dedup             |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| CHỌN MỘT AUDIO VIEW CHO MỖI LƯỢT TRAIN                       |
+--------------------------------------------------------------+
                                |
            +-------------------+------------+
            |                                |
            v                                v
+-----------------------+  +-----------------------------------+
| 50% CLEAN             |  | 50% AUGMENTATION                  |
| Giữ audio gốc         |  | Speed: 0.9 / 1.0 / 1.1            |
|                       |  | Volume: -6 đến +6 dB              |
|                       |  | Noise: SNR 15-30 dB               |
|                       |  | Mỗi phép được chọn p = 0,5        |
+-----------------------+  +-----------------------------------+
            |                                |
            +-------------------+------------+
                                |
                                v
+--------------------------------------------------------------+
| WHISPER PROCESSOR                                            |
| Audio 16 kHz -> 80-bin log-Mel                               |
| Transcript của lượt tương ứng -> token labels                |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| PHOWHISPER-LARGE + LORA                                      |
| Base weights đóng băng; LoRA A/B được cập nhật               |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| LOSS -> ADAMW -> CẬP NHẬT LORA                               |
| Budget cố định, không Validation                             |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| FINAL CHECKPOINT -> MERGE -> EXPORT FP16                     |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| CHẠY ASR TRÊN AUDIO GỐC CỦA TOÀN BỘ TRAIN                    |
| Xử lý current và từng user-turn trong history                |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| OUTPUT: 1 TRANSCRIPT RECORD / SAMPLE                         |
| Text current + history user -> fine-tune LLM                 |
+--------------------------------------------------------------+
```

**1.545 là số source sample, không phải số audio.** Một sample có thể có nhiều
user-turn. Con số 5.036 là tổng 4.083 lượt Fit và 953 lượt Validation cũ trước
audit/dedup. Run mới nhập cả hai phần vào Train; output manifest phải ghi số source
và utterance thực tế. Group có thể chứa nhiều source sample: không gọi 1.545 là
số group nếu chưa đo.

## 2. Đầu vào cần chuẩn bị

Thư mục ban đầu có thể chỉ chứa `ASR.md` và dataset. Để **train thật**, cần thêm:

1. **Dataset Train:** `train.jsonl` cùng toàn bộ WAV được tham chiếu; không dùng
   Public/Private Test cho training, augmentation hay chọn checkpoint.
2. **Base snapshot:** `vinai/PhoWhisper-large`, revision
   `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`, gồm weights, config, tokenizer và
   processor. Sao chép snapshot đã có hoặc tải riêng có chủ đích và ghi revision,
   hash. Chỉ có dataset thì chưa đủ để load model.
3. **Noise assets:** tập WAV non-speech bên ngoài dataset competition, được review
   và có license; kèm manifest/hash. Nếu thiếu noise, vẫn viết được code nhưng
   chưa thể chạy đúng policy E4 đầy đủ.
4. **Môi trường:** Python, Torch/CUDA, Transformers, PEFT, NumPy và GPU tương thích.
   Các version xuất phát từ project cũ là Python 3.11, Torch 2.9.0 với CUDA wheel
   phù hợp, Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.10.1, NumPy 2.3.4 và
   huggingface-hub 0.36.0. Đây là candidate pins; cần tạo environment mới, khóa
   version và kiểm tra tương thích thực tế trên máy đích trước full training.

Snapshot pretrained dùng để **khởi tạo lại** run này. Export E4 đã merge chỉ thích
hợp làm model inference tham chiếu; dùng export ấy làm base sẽ là warm-start và
tạo một phương pháp khác.

Có thể đặt dataset ở vị trí khác bằng `data_root`; các đường dẫn audio trong JSONL
được giải quyết tương đối với thư mục `train/`.

## 3. Hợp đồng dữ liệu

Mỗi dòng của `data/train/train.jsonl` là một JSON object cho một source sample.
Ví dụ cấu trúc rút gọn sau chỉ minh họa các trường ASR cần đọc:

```json
{"id":"sample_001","audio":"audio/current.wav","audio_text":"Nhắc tôi họp lúc tám giờ","history":[{"role":"user","audio":"audio/earlier.wav","audio_text":"Tôi có lịch gì hôm nay"},{"role":"assistant","text":"Bạn có một cuộc họp."}]}
```

Chiếu từng source thành các row ASR:

- Lấy `id` của source; current dùng chính `audio` và `audio_text` của object đó.
- Duyệt `history` theo thứ tự gốc; với mỗi object `role=user`, lấy `audio` và
  `audio_text` của **chính turn đó**. Các role khác không tạo row ASR.
- Gán `turn_id=current` hoặc `history_000`, `history_001`… theo **index gốc
  trong history**, kể cả các vị trí assistant. `utterance_id` ổn định là
  `<sample_id>/<turn_id>`.
- Mỗi row output có ít nhất `sample_id`, `turn_id`, `utterance_id`, `audio`,
  `reference`, `wav_sha256`, `pcm_sha256` và `duration_seconds`. Ví dụ:
  `{"sample_id":"sample_001","turn_id":"current","utterance_id":"sample_001/current","audio":"audio/current.wav","reference":"Nhắc tôi họp lúc tám giờ"}`.
  Các hash/duration được tính từ file thật khi chuẩn bị manifest.
- Chỉ cho phép path WAV nằm dưới `train/audio/`; kiểm tra path traversal và
  symlink sau khi resolve. Không đọc, index hay sao chép `*_environment/`,
  `*_enviroment/`, state ẩn hoặc gold tool calls.
- Audio đầu vào là WAV PCM mono 16-bit, 16 kHz, duration `0 < t <= 30` giây.
  Reject file thiếu, payload bị cắt, audio rỗng/NaN hoặc transcript rỗng; ghi ID
  và lý do. Không crop audio, truncate transcript hoặc bỏ qua row âm thầm.
- Giữ transcript gốc, chỉ normalize Unicode NFC và trim hai đầu khi tokenize.
  Giữ chữ hoa, dấu tiếng Việt, số và từ tiếng Anh; không dùng chuẩn hóa WER làm
  target training. Nếu cùng PCM có transcript mâu thuẫn, dừng để review.
- Mặc định giữ mọi row hợp lệ, kể cả bản trùng, để phản ánh toàn bộ source. Nếu
  quyết định dedup PCM + reference, ghi policy, ID đại diện, provenance và số
  row đã loại trong run manifest; không xóa dữ liệu nguồn.

Identity tham khảo của `train.jsonl` dự án cũ là SHA-256
`24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8`.
Kiểm tra hash, số source và số utterance trước khi dùng các con số 1.545/5.036;
nếu khác, ghi dataset identity và điều tra, không tự giả định đây là cùng dataset.
Run full-data không cần tái tạo split 80/20.

## 4. Chính sách augmentation

Mỗi lần thăm một utterance trong epoch, khởi tạo RNG xác định từ chuỗi UTF-8
`<root_seed>:<epoch>:<utterance_id>:augmentation-v1`. Lấy SHA-256, dùng 8 byte
đầu theo little-endian làm seed cho `numpy.random.default_rng`. Đọc quyết định
ngẫu nhiên theo đúng thứ tự sau để có thể tái lập:

1. Nếu giá trị đầu tiên >= 0,5, dùng **clean**. Nếu < 0,5, vào augmentation branch.
2. Trong branch, lần lượt thử **speed**, **volume**, **noise**; mỗi phép dùng
   một lượt chọn xác suất 0,5 độc lập.
3. Speed chọn đều `0.9 / 1.0 / 1.1`; resample bằng interpolation với count
   `round(len(audio)/speed)`, giữ sample rate 16 kHz. Nếu vượt 480.000 samples,
   dùng factor 1.0 và ghi fallback.
4. Volume chọn gain dB đều từ `-6` đến `+6`, đổi sang hệ số `10**(dB/20)`;
   giới hạn gain để peak không vượt 0,99.
5. Noise chọn từ manifest, offset ngẫu nhiên và lặp noise nếu cần; chọn SNR đều
   trong `15–30 dB`. Với công suất trung bình `P_signal` và `P_noise > 0`,
   hệ số noise là `sqrt(P_signal / (P_noise * 10**(SNR/10)))`.
6. Nếu audio sau biến đổi rỗng, dài hơn 30 giây, không finite hoặc peak >= 1,
   fallback về waveform sạch và log lý do; giữ nguyên transcript và row.

Manifest noise có thể là JSON array với mỗi phần tử gồm `path`, `sha256`,
`license`, `reviewed: true`, `contains_speech: false`. Resolve path an toàn,
kiểm tra license/review/hash và định dạng audio trước khi train. Thiếu noise hợp
lệ làm preflight thất bại; không âm thầm biến thành thí nghiệm không có noise.

**50/50 là tỷ lệ chọn nhánh theo xác suất**, không phải bảo đảm đúng một nửa
row trong từng epoch. Trong augmentation branch có thể không phép nào được chọn,
hoặc speed bằng 1.0, nên số waveform thực sự thay đổi có thể dưới 50%. Không nhân
đôi số row thành clean + aug; không dùng transcript mới cho audio đã augment.

## 5. Kiến trúc model và tham số khởi tạo

PhoWhisper-large là Whisper encoder–decoder. Model được load từ snapshot base,
có 32 encoder layers, 32 decoder layers, `d_model=1280`, 80 mel bins và
`1,543,304,960` logical parameters theo metadata export cũ. Base weights frozen.
Ở mỗi linear layer được chọn, LoRA thêm residual:

```text
output = W_base(input) + (alpha / rank) * B(A(dropout(input)))
         W_base frozen                     A/B trainable
```

| Tham số | Giá trị cho thiết kế độc lập |
|---|---|
| Base / revision | `vinai/PhoWhisper-large` / `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb` |
| Seed | `42` |
| LoRA | `rank=8`, `alpha=16`, `dropout=0.05`, `bias=none`, không `modules_to_save` |
| LoRA initialization | Adapter A/B mới, PEFT default initializer; seed 42. Không load adapter E4 cũ làm điểm bắt đầu |
| Target scope | Cần khóa trong config trước training: `decoder` hoặc `encoder_decoder`; chi tiết bên dưới |
| Precision | `auto`: BF16 nếu GPU/kernel probe đạt, nếu không FP16 + GradScaler; ghi dtype thực tế |
| Micro-batch / accumulation | `1` utterance / `16`; partial window cuối dùng số row thực tế |
| Optimizer | AdamW cho A/B; LR `1e-4`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0.01` |
| Schedule / clip | Linear decay, warm-up `5%` tổng updates; max gradient norm `1.0` |
| Gradient checkpointing | Bật khi train; `use_cache=False`; ưu tiên non-reentrant nếu stack hỗ trợ |
| Epoch budget | Giá trị khởi đầu đề xuất `10` epoch cố định; khóa trước khi chạy |
| Checkpoint | Mỗi `50` updates và cuối run; lưu final, không early-stop hoặc chọn best theo Validation |
| Export / inference | Merge base + adapter, export FP16 đầy đủ; inference offline greedy, `language=vi`, `task=transcribe`, `num_beams=1`, `max_new_tokens=440` |

**LoRA scope là điểm chưa xác nhận của E4 cũ.** Xây code hỗ trợ cả hai giá trị và
yêu cầu config chỉ rõ trước khi train; không tự nhận một scope là scope của E4 cũ:

- `decoder`: `model.decoder.layers.<n>.self_attn.{q_proj,v_proj}` và
  `model.decoder.layers.<n>.encoder_attn.{q_proj,v_proj}`.
- `encoder_decoder`: các target decoder ở trên cộng
  `model.encoder.layers.<n>.self_attn.{q_proj,v_proj}`.

Có thể dùng `decoder` làm cấu hình khởi đầu cho **run mới** nếu chưa có bằng chứng
control cũ; ghi đây là lựa chọn của run mới. Nếu cần tái hiện chính xác E4 cũ,
tìm `resolved-config.json`/run artifact gốc để xác nhận scope trước. Resolve tên
module đầy đủ, assert có đúng target; không dùng tên rời `q_proj`/`v_proj` vì
có thể gắn adapter ngoài phạm vi. Chỉ A/B có `requires_grad=True`.

Đây là LoRA thông thường, không tự bật QLoRA/NF4, bitsandbytes hay đổi rank khi
gặp OOM. Khi cần profile khác, tạo config/run identity khác.

Load `WhisperProcessor` và `WhisperForConditionalGeneration` từ snapshot local;
dùng PEFT generic wrapper nhận `input_features`, không nhầm với wrapper LLM nhận
`input_ids`. Mỗi epoch shuffle các row bằng seed `42 + epoch`; mỗi row nhận một
view theo chính sách augmentation ở trên.

Với 5.036 row không dedup, micro-batch 1, accumulation 16 và 10 epoch đầy đủ,
ước tính `ceil(5036/16) * 10 = 3.150` optimizer updates nếu partial window cuối
mỗi epoch được cập nhật. Đây là ước tính; tính lại sau audit/dedup và khóa budget
trước khi tạo scheduler. Mỗi loss được chia cho số row thực tế của window.

Processor dùng prefix `language=vi, task=transcribe, predict_timestamps=False`;
tránh lặp decoder-start token, từ chối label quá `447` token sau khi bỏ token đó
và mask padding labels bằng `-100`. Các model/processor được load từ snapshot
local với `trust_remote_code=False`.

