"""Offline membership-inference risk auditing for tabular classifiers."""

from .audit import AuditResult, audit_membership

__all__ = ["AuditResult", "audit_membership"]
