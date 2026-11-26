; Inno Setup 6 installer script for Audio8-VC (Windows x64)
; Build via:
;   ISCC.exe /DAppVersion=0.1.0 /DSourceDir=..\..\dist\Audio8-VC /DOutputDir=..\..\dist\release /DVariant=cpu installer.iss

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\Audio8-VC"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist\release"
#endif
#ifndef Variant
  #define Variant "cpu"
#endif

#define AppName "Audio8-VC"
#define AppPublisher "Audio8-VC"
#define AppExeName "Audio8-VC.exe"
#define AppCliName "a8vc.exe"
#define AppURL "https://github.com/Edge0-AI/Audio8_TTS"

[Setup]
AppId={{8E9A7B35-4E81-4B66-9A2E-6F20B8A0810B}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
; Non-admin per-user install by default (no UAC needed), with option for all-users
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
#if VER >= 0x06030000
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#else
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
#endif
CloseApplications=yes
CloseApplicationsFilter=Audio8-VC.exe,a8vc.exe
OutputDir={#OutputDir}
OutputBaseFilename=Audio8-VC-{#AppVersion}-win64-{#Variant}-setup
SetupIconFile=assets\audio8_vc.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
#if FileExists(CompilerPath + "Languages\ChineseSimplified.isl")
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "associate"; Description: "Associate .a8voice voice pack files with Audio8-VC"; GroupDescription: "File associations:"; Flags: checkedonce

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"
Name: "{autoprograms}\{#AppName} (CLI Prompt)"; Filename: "{cmd}"; Parameters: "/k ""set PATH={app};%PATH%& a8vc --help"""; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Register `.a8voice` double-click file association (HKA = HKCU for per-user, HKLM for all-users)
Root: HKA; Subkey: "Software\Classes\.a8voice"; ValueType: string; ValueName: ""; ValueData: "Audio8VC.VoicePack"; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Audio8VC.VoicePack"; ValueType: string; ValueName: ""; ValueData: "Audio8-VC Portable Voice Pack"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Audio8VC.VoicePack\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExeName},0"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Audio8VC.VoicePack\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" ""%1"""; Tasks: associate

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function IsValidWebView2Version(const Ver: String): Boolean;
begin
  Result := (Ver <> '') and (Ver <> '0.0.0.0');
end;

function HasWebView2Runtime(): Boolean;
var
  Ver: String;
begin
  Result :=
    (RegQueryStringValue(HKLM64, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Ver) and IsValidWebView2Version(Ver)) or
    (RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Ver) and IsValidWebView2Version(Ver)) or
    (RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Ver) and IsValidWebView2Version(Ver));
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (not HasWebView2Runtime()) then
  begin
    if not WizardSilent() then
    begin
      MsgBox(
        'Microsoft Edge WebView2 Runtime was not detected on this system.' + #13#10 +
        'Audio8-VC will automatically fall back to your default browser, or you can install WebView2 Runtime from Microsoft for a native desktop window.',
        mbInformation, MB_OK
      );
    end;
  end;
end;
