"""Add source health fields and system settings

Revision ID: 6f8f7af2e5a8
Revises: 19dfa3cccd5d
Create Date: 2026-02-21 15:55:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6f8f7af2e5a8'
down_revision = '19dfa3cccd5d'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'system_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(length=128), nullable=False),
        sa.Column('value_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key'),
    )

    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.add_column(sa.Column('last_success_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('last_failure_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('consecutive_successes', sa.Integer(), server_default='0', nullable=True))
        batch_op.add_column(sa.Column('consecutive_failures', sa.Integer(), server_default='0', nullable=True))
        batch_op.add_column(sa.Column('total_failures', sa.Integer(), server_default='0', nullable=True))
        batch_op.add_column(sa.Column('avg_latency_ms', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('last_error', sa.String(length=512), nullable=True))
        batch_op.add_column(sa.Column('auto_disabled_until', sa.DateTime(timezone=True), nullable=True))
        batch_op.create_index('ix_sources_auto_disabled_until', ['auto_disabled_until'], unique=False)


def downgrade():
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.drop_index('ix_sources_auto_disabled_until')
        batch_op.drop_column('auto_disabled_until')
        batch_op.drop_column('last_error')
        batch_op.drop_column('avg_latency_ms')
        batch_op.drop_column('total_failures')
        batch_op.drop_column('consecutive_failures')
        batch_op.drop_column('consecutive_successes')
        batch_op.drop_column('last_failure_at')
        batch_op.drop_column('last_success_at')

    op.drop_table('system_settings')

