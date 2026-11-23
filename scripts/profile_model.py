import torch
import hydra
import nn_bowl
from torch.profiler import profile, ProfilerActivity, schedule

@hydra.main(version_base=None, config_name='config', config_path=nn_bowl.paths.CONFIG_PATH)
def profile_model(cfg):
    nn_bowl.utils.pretty_print_cfg(cfg)
    device = nn_bowl.utils.get_device()
    print("Using device: ", device)
    
    model = nn_bowl.Model(cfg)
    model.to(device)
    model.train()
    
    dataloaders = nn_bowl.utils.get_dataloaders(cfg, train=True)
    train_dataloader = dataloaders['train']
    
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.train.adamw)
    cost_manager = nn_bowl.CostManager(cfg, model)
    
    print("\n" + "="*80)
    print("Starting profiling - this will take a minute...")
    print("="*80 + "\n")
    
    # Warmup
    print("Warming up...")
    data_iter = iter(train_dataloader)
    for _ in range(3):
        batch = next(data_iter)
        x = batch["img"].to(device)
        y = batch["label"].to(device)
        logits = model(x, noise_anneal=0.0)
        loss = cost_manager(logits, y, None, 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Profiling forward + backward pass...")
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(10):  # Profile 10 batches
            batch = next(data_iter)
            x = batch["img"].to(device)
            y = batch["label"].to(device)
            
            logits = model(x, noise_anneal=0.0)
            loss = cost_manager(logits, y, None, 0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top operations by CUDA time")
    print("="*80 + "\n")
    
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=30,
        max_name_column_width=60
    ))
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top operations by CPU time")
    print("="*80 + "\n")
    
    print(prof.key_averages().table(
        sort_by="cpu_time_total", 
        row_limit=30,
        max_name_column_width=60
    ))
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Memory usage")
    print("="*80 + "\n")
    
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage", 
        row_limit=20,
        max_name_column_width=60
    ))
    
    # Export trace for visualization (optional)
    print("\nExporting trace to profile_trace.json...")
    prof.export_chrome_trace("profile_trace.json")
    print("You can view this in Chrome at chrome://tracing")
    
    print("\n" + "="*80)
    print("Profiling complete!")
    print("="*80)

if __name__ == '__main__':
    profile_model()