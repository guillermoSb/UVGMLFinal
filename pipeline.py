import kfp
from google.cloud import aiplatform

from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import ModelUndeployOp
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp
from google_cloud_pipeline_components.v1.model import ModelGetOp




project_id = "mlops2024-441121"
location = "us-central1"
pipeline_root="gs://mlops2024-pipeline/pipe-runs"

from kfp.v2.dsl import component, Input, Output, Artifact
from kfp.v2.dsl import component, OutputPath


@component(
		packages_to_install=['google-cloud-aiplatform'],
		base_image='python:3.8',
)
def undeploy_all_models_from_endpoint(
		project_id: str,
		location: str,
		endpoint_resource_name: str,
):
		from google.cloud import aiplatform
		aiplatform.init(project=project_id, location=location)
		
		endpoint = aiplatform.Endpoint(endpoint_resource_name)
		deployed_models = endpoint.list_models()
		
		for deployed_model in deployed_models:
				print(f"Undeploying model: {deployed_model.model}")
				endpoint.undeploy(
						deployed_model_id=deployed_model.id,
						sync=True,
				)


@component
def log_model_upload_output(model_resource: Input[Artifact]):
		print("Model resource URI:", model_resource.uri)
		print("Model resource NAME:", model_resource.name)
		# You can add more logs if needed

@component(
		packages_to_install=['google-cloud-aiplatform'],
		base_image='python:3.8',
)
def upload_model_custom(
		project_id: str,
		location: str,
		display_name: str,
		artifact_uri: str,
		serving_container_image_uri: str,
		parent_model: str = '',
		model_resource: Output[Artifact] = None,
):
		from google.cloud import aiplatform
		aiplatform.init(project=project_id, location=location)
		model_resource = model_resource or Output[Artifact]()  
		model = aiplatform.Model.upload(
				display_name=display_name,
				artifact_uri=artifact_uri,
				serving_container_image_uri=serving_container_image_uri,
				parent_model=parent_model if parent_model else None,
		)
		model.wait()
		# Output the resource name for downstream components
		print("MODEL RESOURCE NAME: ", model.resource_name)
		print("VERSIONED MODEL RESOURCE NAME: ", model.versioned_resource_name)
		model_resource.uri = model.versioned_resource_name
		
		
		


@kfp.dsl.pipeline(name="marathontime-training-deployment-pipeline", pipeline_root=pipeline_root)
def marathontime_pipeline():
		
		# Define the training job step - will be used for a scikit-learn model
		training_job = CustomTrainingJobOp(
				project=project_id,
				location=location,
				display_name="marathontime-training",
				worker_pool_specs=[
						{
								"machine_spec": {
										"machine_type": "n1-standard-4",
								},
								"replica_count": 1,
								"container_spec": {
										"image_uri": "us-central1-docker.pkg.dev/mlops2024-441121/marathontime/marathontime:latest",
								},
						}
				],
				base_output_directory="gs://marathon-time/model-output/",
		)
		existing_model_resource_name = f"projects/{project_id}/locations/{location}/models/5526661112155602944"
		# Model upload step
		model_upload_op = upload_model_custom(
		project_id=project_id,
		location=location,
		display_name="marathontime",
		artifact_uri='gs://marathon-time/model-output/',
		serving_container_image_uri="us-docker.pkg.dev/cloud-aiplatform/prediction/sklearn-cpu.1-3:latest",
		parent_model=existing_model_resource_name,  # Include if updating an existing model
		)	
		model_upload_op.after(training_job)
		# **Add Logging Step Here**
		log_model_op = log_model_upload_output(
				model_resource=model_upload_op.outputs['model_resource']
		)
		log_model_op.after(model_upload_op)
		# Retrieve the versioned model resource name
		get_model_version_op = ModelGetOp(
				project=project_id,
				model_name='5526661112155602944',
		)
		get_model_version_op.after(model_upload_op)
		
		create_endpoint_op = EndpointCreateOp(
				project=project_id,
				location=location,
				display_name = "test-endpoint",
				labels={"test": "test"},
		)
		create_endpoint_op.after(get_model_version_op)
		
		undeploy_op = undeploy_all_models_from_endpoint(
				project_id=project_id,
				location=location,
				endpoint_resource_name=create_endpoint_op.outputs["endpoint"],
		)
		undeploy_op.after(create_endpoint_op)
		# LOG MODEL UPLOAD OUTPUT MODEL
		# Model deploy step
		print(create_endpoint_op.outputs)
		model_deploy_op = ModelDeployOp(
				model=get_model_version_op.outputs["model"],
				endpoint=create_endpoint_op.outputs["endpoint"],
				dedicated_resources_machine_type="n1-standard-4",
				dedicated_resources_min_replica_count=1,
		)
		model_deploy_op.after(undeploy_op)
	

		
		
		

		
from kfp.v2 import compiler

# Compile the pipeline
compiler.Compiler().compile(
		pipeline_func=marathontime_pipeline,
		package_path="marathontime_pipeline.json"
)