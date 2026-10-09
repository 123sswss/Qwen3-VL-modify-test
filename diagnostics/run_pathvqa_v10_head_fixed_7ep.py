"""Independent authorized seven-epoch run; fixed epoch7 primary, no selection."""
from diagnostics.run_pathvqa_v10_head_fixed import main

if __name__ == "__main__":
    raise SystemExit(main(seven_epochs=True))
