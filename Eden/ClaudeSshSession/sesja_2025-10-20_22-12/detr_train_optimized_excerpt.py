        prefetch_factor=2 if args.num_workers > 0 else None,  # Optymalizacja
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    # Load model
    if rank == 0:
        print(f"Loading model configuration from checkpoint: {args.model_checkpoint}")
    
    try:
        config = DetrConfig.from_pretrained(
            args.model_checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        config.num_queries = args.num_queries
        
        model = DetrForObjectDetection.from_pretrained(
            args.model_checkpoint,
            config=config,
            ignore_mismatched_sizes=True
        ).to(device)
        
        # Compile model dla dodatkowej wydajności (PyTorch 2.0+)
        # DISABLED DUE TO DETR SIGSEGV BUG: if hasattr(torch, 'compile') and args.compile_model:
        # DISABLED DUE TO DETR SIGSEGV BUG:     if rank == 0:
        # DISABLED DUE TO DETR SIGSEGV BUG:         print("Compiling model with torch.compile...")
        # DISABLED DUE TO DETR SIGSEGV BUG:     model = torch.compile(model, mode="reduce-overhead")
        
        # Wrap model in DDP
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False  # Optymalizacja - ustaw True tylko jeśli potrzebne
        )
        
        if rank == 0:
            print(f"Model loaded successfully with DDP on GPU {local_rank}")
            
    except Exception as e:
        if rank == 0:
            print(f"Error loading model: {e}")
        cleanup_ddp()
        return

    # Setup optimizer
    try:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            fused=True if torch.cuda.is_available() else False  # Fused optimizer dla lepszej wydajności
        )
    except Exception as e:
        if rank == 0:
            print(f"Error setting up optimizer: {e}")
        cleanup_ddp()
        return

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=args.use_amp)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_training:
        latest_checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint_path:
            # Note: pass the base model (not DDP wrapped) for loading
            model.module, optimizer, scaler, start_epoch = load_checkpoint(
                latest_checkpoint_path, model.module, optimizer, scaler, device
            )

    # Training loop
    if rank == 0:
        print(f"Starting training loop from epoch {start_epoch}...")
        print(f"Mixed Precision Training: {'Enabled' if args.use_amp else 'Disabled'}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
