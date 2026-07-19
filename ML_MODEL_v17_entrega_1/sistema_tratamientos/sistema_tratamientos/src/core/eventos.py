from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DecisionAuditada:
    timestamp:     datetime
    fase:          str
    motor:         str
    accion:        str
    justificacion: str
    datos_usados:  dict
    regla_id:      str | None = None
    usuario_id:    str | None = None
    paciente_id:   str           = ""

    def a_dict(self) -> dict:
        return {
            "timestamp":     self.timestamp.isoformat(),
            "fase":          self.fase,
            "motor":         self.motor,
            "accion":        self.accion,
            "justificacion": self.justificacion,
            "datos_usados":  self.datos_usados,
            "regla_id":      self.regla_id,
            "usuario_id":    self.usuario_id,
            "paciente_id":   self.paciente_id,
        }
