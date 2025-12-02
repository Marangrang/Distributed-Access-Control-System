"""Tests for Airflow DAGs."""
import pytest
from airflow.models import DagBag
from datetime import datetime
import os


@pytest.fixture(scope="module")
def dagbag():
    """Load all DAGs for testing."""
    dags_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'airflow',
        'dags'
    )
    return DagBag(dag_folder=dags_folder, include_examples=False)


class TestDAGIntegrity:
    """Test DAG integrity and basic properties."""

    def test_dag_loading(self, dagbag):
        """Test that all DAGs load without errors."""
        assert len(dagbag.import_errors) == 0, \
            f"DAG import errors: {dagbag.import_errors}"

    def test_required_dags_exist(self, dagbag):
        """Test that required DAGs exist."""
        required_dags = [
            'face_data_ingestion_pipeline',
            'face_recognition_training_pipeline'
        ]
        for dag_id in required_dags:
            assert dag_id in dagbag.dags, f"DAG {dag_id} not found"

    def test_dag_has_tags(self, dagbag):
        """Test that all DAGs have tags."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.tags, f"DAG {dag_id} has no tags"

    def test_dag_has_owner(self, dagbag):
        """Test that all DAGs have an owner."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.owner, f"DAG {dag_id} has no owner"
            assert dag.owner != 'airflow', \
                f"DAG {dag_id} uses default 'airflow' owner"

    def test_dag_has_retries(self, dagbag):
        """Test that all DAGs have retry configuration."""
        for dag_id, dag in dagbag.dags.items():
            for task in dag.tasks:
                assert task.retries is not None, \
                    f"Task {task.task_id} in DAG {dag_id} has no retries"

    def test_dag_catchup_disabled(self, dagbag):
        """Test that catchup is disabled for all DAGs."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.catchup is False, \
                f"DAG {dag_id} has catchup enabled"

    def test_dag_max_active_runs(self, dagbag):
        """Test that max_active_runs is set."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.max_active_runs is not None, \
                f"DAG {dag_id} has no max_active_runs set"


class TestTrainingPipelineDAG:
    """Test face_recognition_training_pipeline DAG."""

    @pytest.fixture
    def dag(self, dagbag):
        """Get training pipeline DAG."""
        return dagbag.dags.get('face_recognition_training_pipeline')

    def test_dag_exists(self, dag):
        """Test that training pipeline DAG exists."""
        assert dag is not None

    def test_dag_task_count(self, dag):
        """Test that DAG has expected number of tasks."""
        assert len(dag.tasks) == 9, \
            f"Expected 9 tasks, found {len(dag.tasks)}"

    def test_dag_has_required_tasks(self, dag):
        """Test that DAG has all required tasks."""
        required_tasks = [
            'start',
            'fetch_images_from_minio',
            'preprocess_images',
            'extract_faces',
            'generate_embeddings',
            'build_faiss_index',
            'save_index_to_minio',
            'update_database',
            'end'
        ]
        task_ids = [task.task_id for task in dag.tasks]
        for task_id in required_tasks:
            assert task_id in task_ids, \
                f"Required task {task_id} not found in DAG"

    def test_dag_task_dependencies(self, dag):
        """Test task dependencies are correct."""
        start_task = dag.get_task('start')
        fetch_task = dag.get_task('fetch_images_from_minio')
        end_task = dag.get_task('end')

        # Start task should have downstream dependencies
        assert len(start_task.downstream_task_ids) > 0

        # Fetch task should come after start
        assert 'start' in [t.task_id for t in fetch_task.upstream_list]

        # End task should have upstream dependencies
        assert len(end_task.upstream_task_ids) > 0


class TestIngestionPipelineDAG:
    """Test face_data_ingestion_pipeline DAG."""

    @pytest.fixture
    def dag(self, dagbag):
        """Get ingestion pipeline DAG."""
        return dagbag.dags.get('face_data_ingestion_pipeline')

    def test_dag_exists(self, dag):
        """Test that ingestion pipeline DAG exists."""
        assert dag is not None

    def test_dag_task_count(self, dag):
        """Test that DAG has expected number of tasks."""
        assert len(dag.tasks) == 3, \
            f"Expected 3 tasks, found {len(dag.tasks)}"

    def test_dag_has_required_tasks(self, dag):
        """Test that DAG has all required tasks."""
        required_tasks = [
            'start',
            'ingest_data',
            'end'
        ]
        task_ids = [task.task_id for task in dag.tasks]
        for task_id in required_tasks:
            assert task_id in task_ids, \
                f"Required task {task_id} not found in DAG"

    def test_dag_schedule(self, dag):
        """Test that DAG has correct schedule."""
        # Ingestion should run daily
        assert dag.schedule_interval is not None, \
            "DAG should have a schedule_interval"
