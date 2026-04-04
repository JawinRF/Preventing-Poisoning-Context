import java.util.Properties

plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.detekt)
    alias(libs.plugins.ktlint)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.openclaw.android"
    compileSdk = 36

    dependenciesInfo {
        includeInApk = false
        includeInBundle = false
    }

    defaultConfig {
        applicationId = "com.openclaw.android"
        minSdk = 24
        //noinspection ExpiredTargetSdkVersion
        targetSdk = 28
        versionCode = 10
        versionName = "1.0.0"

        ndk { abiFilters += listOf("arm64-v8a") }

        buildConfigField(
            "String",
            "BOOTSTRAP_URL",
            "\"https://github.com/termux/termux-packages/releases/download/bootstrap-2026.02.12-r1%2Bapt.android-7/bootstrap-aarch64.zip\"",
        )
        buildConfigField(
            "String",
            "WWW_URL",
            "\"https://github.com/AidanPark/openclaw-android-app/releases/download/v1.0.0/www.zip\"",
        )
        buildConfigField(
            "String",
            "CONFIG_URL",
            "\"https://raw.githubusercontent.com/AidanPark/openclaw-android-app/main/config.json\"",
        )
    }

    signingConfigs {
        create("release") {
            val props = project.rootProject.file("local.properties")
            if (props.exists()) {
                val localProps = Properties().apply { props.inputStream().use { load(it) } }
                val storePath = localProps.getProperty("RELEASE_STORE_FILE", "")
                if (storePath.isNotEmpty()) {
                    storeFile = file(storePath)
                    storePassword = localProps.getProperty("RELEASE_STORE_PASSWORD", "")
                    keyAlias = localProps.getProperty("RELEASE_KEY_ALIAS", "")
                    keyPassword = localProps.getProperty("RELEASE_KEY_PASSWORD", "")
                }
            }
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-DEBUG"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    @Suppress("UnstableApiUsage")
    testOptions {
        unitTests.all { it.useJUnitPlatform() }
        unitTests.isReturnDefaultValues = true
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    packaging {
        jniLibs { useLegacyPackaging = true }
        resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
    }

    aaptOptions {
        noCompress("onnx")
    }
}

dependencies {
    // Terminal emulation (Termux-forked)
    implementation(project(":terminal-emulator"))
    implementation(project(":terminal-view"))

    // AndroidX core
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.gson)

    // PRISM security: Room database for audit logging
    implementation(libs.room.runtime)
    implementation(libs.room.ktx)
    ksp(libs.room.compiler)

    // PRISM security: ONNX Runtime for on-device ML (Layer 2)
    implementation(libs.onnxruntime.android)

    // PRISM security: NanoHTTPD for HTTP sidecar on port 8766
    implementation(libs.nanohttpd)

    // Test dependencies
    testImplementation(libs.junit5)
    testRuntimeOnly(libs.junit5.platform.launcher)
    testImplementation(libs.mockk)
    testImplementation(libs.kotlinx.coroutines.test)
}

// --- www build automation ---
val wwwProjectDir = file("$rootDir/www")
val assetsWwwDir = file("$projectDir/src/main/assets/www")

val buildWww by tasks.registering(Exec::class) {
    description = "Build React UI (npm run build)"
    group = "build"
    workingDir = wwwProjectDir
    commandLine("npm", "run", "build")
    inputs.dir(wwwProjectDir.resolve("src"))
    inputs.files(
        wwwProjectDir.resolve("package.json"),
        wwwProjectDir.resolve("tsconfig.json"),
        wwwProjectDir.resolve("vite.config.ts"),
    )
    outputs.dir(wwwProjectDir.resolve("dist"))
}

val syncWwwAssets by tasks.registering(Sync::class) {
    description = "Copy React dist/ into assets/www/"
    group = "build"
    dependsOn(buildWww)
    from(wwwProjectDir.resolve("dist"))
    into(assetsWwwDir)
}

tasks.named("preBuild") {
    dependsOn(syncWwwAssets)
}

detekt {
    buildUponDefaultConfig = true
    allRules = false
    config.setFrom("$rootDir/detekt.yml")
}

tasks.withType<io.gitlab.arturbosch.detekt.Detekt>().configureEach {
    jvmTarget = "17"
    reports {
        html.required.set(true)
        sarif.required.set(true)
        xml.required.set(false)
        txt.required.set(false)
    }
}

tasks.withType<io.gitlab.arturbosch.detekt.DetektCreateBaselineTask>().configureEach {
    jvmTarget = "17"
}

configure<org.jlleitschuh.gradle.ktlint.KtlintExtension> {
    android.set(true)
    outputToConsole.set(true)
    ignoreFailures.set(false)
    filter {
        exclude("**/generated/**")
    }
}
