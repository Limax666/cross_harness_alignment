#!/usr/bin/env bash
# per-PID RssShmem/rss prober for the r13 pageable-offload validation
RUN="$1"
echo -e "time\tpid\tcomm\trss_gb\tshm_gb" > "$RUN/mem_probe.tsv"
for i in $(seq 1 720); do
  ts=$(date +%H:%M:%S)
  for p in $(pgrep -f 'ray::WorkerDict' 2>/dev/null); do
    c=$(awk '/^(VmRSS|RssShmem):/{printf "%s ",$2}' /proc/$p/status 2>/dev/null)
    [ -n "$c" ] && echo -e "$ts\t$p\tWorkerDict\t$c" | awk -F'\t' '{printf "%s\t%s\t%s\t%.1f\t%.1f\n",$1,$2,$3,$4/1048576,$5/1048576}' >> "$RUN/mem_probe.tsv"
  done
  echo -e "$ts\tAVAIL\t-\t$(awk '/MemAvailable:/{print $2/1048576}' /proc/meminfo)\t-" >> "$RUN/mem_probe.tsv"
  sleep 15
done
