-dontwarn com.google.android.play.core.**
-keep class com.google.android.play.core.** { *; }

# llama_flutter_android - JNI, Foreground Service, Kotlin callbacks
-keep class com.write4me.llama_flutter_android.** { *; }
-keep class * extends android.app.Service { *; }
-keepclassmembers class * {
    native <methods>;
}

# Kotlin functions used by JNI (R8 renames these and breaks native calls)
-keep class kotlin.jvm.functions.** { *; }
-keep class kotlin.coroutines.** { *; }
-keep class kotlinx.coroutines.** { *; }
-keepclassmembers class * extends kotlin.coroutines.jvm.internal.BaseContinuationImpl { *; }
-keep class kotlin.Unit { *; }
