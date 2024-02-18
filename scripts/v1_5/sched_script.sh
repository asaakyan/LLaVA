pid=546322

while [ -d "/proc/$pid" ]; do
    sleep 1
done

./finetune_task_lora_vflute.sh