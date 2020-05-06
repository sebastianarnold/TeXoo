#!/bin/sh
mvn clean deploy -Pdeploy-jar -DaltDeploymentRepository=de.datexis.internal::default::https://repository.datexis.com/repository/de.datexis.internal/
# mvn -pl texoo-core clean deploy -Pdeploy-jar -DaltDeploymentRepository=de.datexis.internal::default::http://repository.datexis.com/repository/de.datexis.internal/
# mvn -pl texoo-entity-recognition clean deploy -Pdeploy-jar -DaltDeploymentRepository=de.datexis.internal::default::http://repository.datexis.com/repository/de.datexis.internal/
# mvn -pl texoo-entity-linking clean deploy -Pdeploy-jar -DaltDeploymentRepository=de.datexis.internal::default::http://repository.datexis.com/repository/de.datexis.internal/
# mvn -pl texoo-sector clean deploy -Pdeploy-jar -DaltDeploymentRepository=de.datexis.internal::default::http://repository.datexis.com/repository/de.datexis.internal/
