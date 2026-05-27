pass
import yaml
import logging
from pathlib import Path
from quant.config import config
logger = logging.getLogger(__name__)

def detect_symbol_from_path(data_path: str) -> str:
    pass
    path = Path(data_path)
    for part in path.parent.parts:
        if part in config.MARKET_CONFIGS:
            return part
    return 'ES'

def load_market_config(symbol: str):
    pass
    yaml_path = config.MARKET_CONFIGS.get(symbol)
    if not yaml_path or not Path(yaml_path).exists():
        logger.warning(f'Market config for {symbol} not found at {yaml_path}, using global defaults.')
        return
    with open(yaml_path, 'r') as f:
        market_cfg = yaml.safe_load(f)
    overrides = {'ROLL_WINDOWS': market_cfg.get('roll_windows'), 'ROLL_WINDOWS_1H': market_cfg.get('roll_windows_1h'), 'ROLL_WINDOWS_DAILY': market_cfg.get('roll_windows_daily'), 'REGIME_HIGH_THRESH': market_cfg.get('regime_high_thresh'), 'REGIME_LOW_THRESH': market_cfg.get('regime_low_thresh'), 'HTF_TREND_WINDOWS': market_cfg.get('htf_trend_windows'), 'HTF_VOLATILITY_WINDOWS': market_cfg.get('htf_volatility_windows'), 'SLIPPAGE_K': market_cfg.get('slippage_k'), 'VOL_PENALTY': market_cfg.get('vol_penalty'), 'COMMISSION_PER_TRADE': market_cfg.get('commission_per_trade'), 'MAX_LEVERAGE': market_cfg.get('max_leverage'), 'TARGET_VOL': market_cfg.get('target_vol')}
    for attr, value in overrides.items():
        if value is not None:
            setattr(config, attr, value)
            logger.info(f'Overrode {attr} = {value} for {symbol}')