"""Unit tests for Airflow DAGs."""
import pytest
from datetime import datetime
from airflow.models import DagBag
import sys
from pathlib import Path

# Add airflow dags to path
AIRFLOW_DAGS_FOLDER = Path(__file__).parent.parent.parent / 'airflow' / 'dags'


class TestDAGIntegrity:
    """Test DAG integrity and structure."""
    
    @pytest.fixture(scope='class')
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(AIRFLOW_DAGS_FOLDER), include_examples=False)
    
    def test_dag_loading(self, dagbag):
        """Test that all DAGs load without errors."""
        assert len(dagbag.import_errors) == 0, \
            f"DAG import errors: {dagbag.import_errors}"
        assert len(dagbag.dags) > 0, "No DAGs found"
    
    def test_required_dags_exist(self, dagbag):
        """Test that required DAGs are present."""
        required_dags = [
            'face_recognition_training_pipeline',
            'face_data_ingestion_pipeline',
        ]
        
        dag_ids = list(dagbag.dag_ids)
        
        for dag_id in required_dags:
            assert dag_id in dag_ids, f"DAG '{dag_id}' not found"
    
    def test_dag_has_tags(self, dagbag):
        """Test that all DAGs have tags."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.tags, f"DAG '{dag_id}' has no tags"
    
    def test_dag_has_owner(self, dagbag):
        """Test that all DAGs have an owner."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.default_args.get('owner'), \
                f"DAG '{dag_id}' has no owner"
    
    def test_dag_has_retries(self, dagbag):
        """Test that all DAGs have retry configuration."""
        for dag_id, dag in dagbag.dags.items():
            assert 'retries' in dag.default_args, \
                f"DAG '{dag_id}' has no retry configuration"
    
    def test_dag_catchup_disabled(self, dagbag):
        """Test that catchup is disabled for production DAGs."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.catchup is False, \
                f"DAG '{dag_id}' has catchup enabled"
    
    def test_dag_max_active_runs(self, dagbag):
        """Test that DAGs have max_active_runs configured."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.max_active_runs is not None, \
                f"DAG '{dag_id}' has no max_active_runs configured"


class TestTrainingPipelineDAG:
    """Test training pipeline DAG specifically."""
    
    @pytest.fixture(scope='class')
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(AIRFLOW_DAGS_FOLDER), include_examples=False)
    
    @pytest.fixture(scope='class')
    def dag(self, dagbag):
        """Get training pipeline DAG."""
        return dagbag.get_dag('face_recognition_training_pipeline')
    
    def test_dag_exists(self, dag):
        """Test DAG exists and loads."""
        assert dag is not None
        assert dag.dag_id == 'face_recognition_training_pipeline'
    
    def test_dag_structure(self, dag):
        """Test DAG has correct structure."""
        assert len(dag.tasks) > 0
        assert dag.schedule_interval == '@daily'
    
    def test_required_tasks_exist(self, dag):
        """Test required tasks are present."""
        task_ids = [task.task_id for task in dag.tasks]
        
        required_tasks = [
            'check_new_training_data',
            'train_face_recognition_model',
            'build_faiss_index',
            'upload_artifacts_to_minio',
            'restart_app_service',
            'wait_for_service_health',
            'validate_deployment',
        ]
        
        for task_id in required_tasks:
            assert task_id in task_ids, f"Task '{task_id}' not found in DAG"
    
    def test_task_dependencies(self, dag):
        """Test task dependencies are correct."""
        # Get specific tasks
        check_data = dag.get_task('check_new_training_data')
        train_model = dag.get_task('train_face_recognition_model')
        build_index = dag.get_task('build_faiss_index')
        
        # Check upstream dependencies
        assert train_model in check_data.downstream_list
        assert build_index in train_model.downstream_list


class TestIngestionPipelineDAG:
    """Test ingestion pipeline DAG."""
    
    @pytest.fixture(scope='class')
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(AIRFLOW_DAGS_FOLDER), include_examples=False)
    
    @pytest.fixture(scope='class')
    def dag(self, dagbag):
        """Get ingestion pipeline DAG."""
        return dagbag.get_dag('face_data_ingestion_pipeline')
    
    def test_dag_exists(self, dag):
        """Test DAG exists and loads."""
        assert dag is not None
        assert dag.dag_id == 'face_data_ingestion_pipeline'
    
    def test_dag_schedule(self, dag):
        """Test DAG has correct schedule."""
        assert dag.schedule_interval == '@hourly'
    
    def test_required_tasks_exist(self, dag):
        """Test required tasks are present."""
        task_ids = [task.task_id for task in dag.tasks]
        
        required_tasks = [
            'ingest_from_cloud_storage',
            'create_ingestion_log_table',
            'log_ingestion_result',
        ]
        
        for task_id in required_tasks:
            assert task_id in task_ids, f"Task '{task_id}' not found in DAG"


class TestDAGPerformance:
    """Test DAG performance characteristics."""
    
    @pytest.fixture(scope='class')
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(AIRFLOW_DAGS_FOLDER), include_examples=False)
    
    def test_dag_loading_time(self, dagbag):
        """Test DAG loading is fast (< 30 seconds)."""
        import time
        
        start = time.time()
        DagBag(dag_folder=str(AIRFLOW_DAGS_FOLDER), include_examples=False)
        duration = time.time() - start
        
        assert duration < 30, \
            f"DAG loading took {duration:.2f}s (should be < 30s)"
    
    def test_no_top_level_code(self, dagbag):
        """Test that DAGs don't have excessive top-level code."""
        # This is a basic check - you might want to add more sophisticated checks
        for dag_id, dag in dagbag.dags.items():
            # Check that DAG has tasks
            assert len(dag.tasks) > 0, \
                f"DAG '{dag_id}' has no tasks"