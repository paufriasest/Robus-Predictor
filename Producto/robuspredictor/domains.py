def split_training_domains(x, y, n_domain, verbose=False):
    total_rows = len(x)
    base_size = total_rows // n_domain
    remainder = total_rows % n_domain
    
    domains = []
    
    start = 0
    
    if verbose:
        print("\n[Domains] División interna del dataset de entrenamiento")
        print(f"[Domains] Total registros: {total_rows}")
        print(f"[Domains] Número de dominios: {n_domain}")
        print(f"[Domains] Tamaño base por dominio: {base_size}")
        print(f"[Domains] Registros sobrantes: {remainder}")
    
    for domain_number in range(1, n_domain + 1):
        extra = 1 if domain_number <= remainder else 0
        domain_size = base_size + extra
        end = start + domain_size
        
        X_domain = x.iloc[start:end].copy()
        y_domain = y.iloc[start:end].copy()
        
        domains.append(
            {
                "domain": domain_number,
                "x": X_domain,
                "y": y_domain,
            }
        )
        
        if verbose:
            print(
                f"[Domains] Dominio {domain_number}: "
                f"filas {start} a {end - 1} | registros={len(X_domain)}"
            )
        
        start = end
    
    return domains