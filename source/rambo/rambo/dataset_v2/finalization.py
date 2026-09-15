"""Single-writer close/validate/rename helper, not a recording or publication pipeline."""
from pathlib import Path
import os
from .media import validate_video
from ..contracts_v2.validation import ContractError, require


def finalize_video(partial, final, *, count, fps, tools, profile=None, receipt=None):
    partial, final = Path(partial), Path(final)
    require(partial.name.endswith('.partial.mp4'), 'media', 'partial suffix required')
    require(partial.parent.resolve() == final.parent.resolve(), 'media', 'atomic rename requires same directory')
    require(final.suffix == '.mp4' and not final.name.endswith('.partial.mp4'), 'media', 'final suffix')
    require(not final.exists(), 'media', 'refuse overwrite')
    try:
        report = validate_video(partial, count=count, fps=fps, tools=tools, profile=profile,
                                receipt=receipt, allow_partial=True)
    except ContractError as error:
        # Caller must persist this status in its episode metadata. No repair is attempted.
        return {'episode_status': 'invalid', 'path': str(partial), 'code': error.code, 'detail': error.detail}
    os.rename(partial, final)
    return {'episode_status': 'video_validated', 'path': str(final), 'validation': report}
